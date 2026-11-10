# Tag KV Block 生命周期（实现说明）

本文描述 `tag-kv-block-usage.md` 中 tag 的**实现生命周期**：owner tag 在物理块上
何时写入、何时可见、何时消失，以及请求级 tag 如何驱动这些变化。设计口径以
[`tag-kv-block-usage.md`](tag-kv-block-usage.md) 为准，本文补充实现落点、状态机、
边界行为和已知加固。

## 1. 代码落点

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 物理块 | `src/llama-block-manager.h/.cpp` | `physical_block::tag` 的存储；`set_tag/tag`；ref 归 0 时清除 |
| full 缓存 | `src/llama-kv-cache-paged.h/.cpp` | 分配时打 tag；`collect_tag_blocks()` 遍历 live prefix 项 |
| linear 缓存 | `src/llama-linear-state-cache.h/.cpp` | `seq_state.tag`；checkpoint 打 tag；`collect_tag_blocks()` |
| **tag 聚合/查询** | `src/llama-paged-tag-manager.h/.cpp` | 物化 `owned_full/owned_linear/shared_full`；计数、id、遍历、池统计 |
| 调度器 | `src/llama-paged-scheduler-impl.h/.cpp` | `sequence_group.tag`、`set_request_tag()`、fork 继承；提供活跃请求视图 |
| C API | `include/llama.h`、`src/llama-paged-scheduler.cpp` | `llama_paged_scheduler_*tag*`、`get_kv_cache_stats` |
| 服务 | `tools/server/*` | 请求字段 `tag`、`/kv_stats`、`/tag_usage`、`/slots` 可见性 |

tag 的**唯一存储**是物理块上的 owner tag；`owned/shared` 映射每次查询时物化，
不维护增量状态（避免 refcount 漏删/误删）。

```
 请求 tag ──写入──▶ sequence_group.tag / linear seq_state.tag
                        │
              allocate()│create_checkpoint(GPU)
                        ▼
             physical_block.tag  ──collect_tag_blocks()──▶ owned_*
                        ▲                                  ▲
                        │ release_*: ref==0 时 clear       │
                        │                                  │
             活跃请求 block_table ──────materialize──────▶ shared_full
```

## 2. 生命周期状态机（单个 full 块）

```
[free, tag="" ]
     │  checkout_gpu_blocks + set_tag(T)        （llama_kv_cache_paged::allocate）
     ▼
[allocated, tag=T, 不可见/不计数]                （partial tail、无 hash）
     │  块填满 + register_prefix_block(hash)
     ▼
[reusable, 可见, owned(T)]  ◀──────────────┐
     │                                      │ 命中方 T' 结束，且 block_table 清空
     │  T' 命中: add_ref + take_cached_ref  │ （owner 不变）
     ▼                                      │
[referenced by T'，仍 owned(T)；shared(T') 含该 id]
     │  owner 请求结束 (ref 仍 >0)
     ▼
[cached-resident, owned(T)，等待复用或淘汰]
     │  evict_cached_prefix_block / 最后一次 release（ref==0）
     ▼
[free, tag 清除]  ── generation++，旧索引项自动失效
```

linear checkpoint 的状态机相同，只是“可见”的条件是
`checkpoint_index` 中 live 且 GPU 驻留；ACTIVE state 块永远停在
`[allocated, tag="", 不可见]`。

## 3. 写入 / 可见 / 消失事件表

### 3.1 写入 owner tag

| 事件 | 代码 | 说明 |
| --- | --- | --- |
| full 分配 | `llama_kv_cache_paged::allocate()` | 每个新 checkout 的块 `set_tag(id, group.tag)`；**尚未**可见 |
| linear checkpoint 创建 | `create_checkpoint()` | 仅当 checkpoint 落在 GPU 池时 `set_tag(ckpt_id, seq_state.tag)`；CPU 池不打 |
| linear ACTIVE 分配 | `ensure_active(seq_id, tag)` | 只写 `seq_state.tag`，**不给 ACTIVE 块打 tag** |
| 请求 fork | `fork_request()` | 子请求继承 `group.tag`；`seq_cp` 复制 tag；`fork_checkpoint` 在替换 ACTIVE 块前保存/恢复 tag |
| 请求 tag 注入 | `set_request_tag()` | 仅允许在首次调度前（`!cache_reserved && block_table.empty()`）；若 linear 已 active 同步 `set_seq_tag` |

### 3.2 进入映射（可见 / 计数）

| 事件 | 条件 |
| --- | --- |
| full 块填满并注册 | `register_prefix_block()` 成功，且查询时 `is_gpu(id) && ref>0 && generation 匹配` |
| linear checkpoint | `checkpoint_index.entries()` 中 `checkpoint_is_live(e) && is_gpu(e.block_id)` |
| shared 出现 | 某个带 tag 的活跃请求的 `block_table` 引用了上述可复用块，且物理 owner 不是该 tag |

### 3.3 清除 owner tag / 映射消失

| 事件 | 机制 |
| --- | --- |
| 请求结束、块未保留 | `free_blocks()` → `release_gpu_blocks()`，ref 归 0 → `tag.clear()` |
| cancel / truncate / 直接释放 | 同上（truncate 走 `release_block`） |
| prefix 淘汰 | `evict_cached_prefix_block()` → `release_block()` |
| 同一 hash 链被新物理块重新注册 | `register_prefix_block()` 释放旧的 cached 副本（ref==1 时） |
| **命中方**请求结束 | 只清空其 `block_table`，shared 映射消失；owner 不受影响 |
| linear checkpoint 淘汰 | `evict_checkpoint()` → `release_block()` |
| linear checkpoint 降级 CPU | `demote_checkpoint()`：GPU 块释放（tag 清除），CPU 块不打 tag |
| full / linear swap_out | 视为释放：GPU 块 release（tag 清除） |
| full / linear swap_in | 新 GPU 块**不补 tag**（等价新分配的无 tag 块） |
| `clear()`（full / linear） | 重建 block_manager + 清空索引，全部消失 |
| 旧索引项 generation 不匹配 | 查询过滤，自动不可见 |

### 3.4 不改变 tag 的事件

`add_ref_blocks` / `share_blocks` / `take_cached_ref` / `copy_block_content` /
`register_prefix_block`（同块重注册）/ `release_sequence`（保留块）/
`set_seq_pos`/rollback 等。命中复用只增加引用，不把 owner 改成命中方。

## 4. 请求级 tag 的生命周期

```
add_request ──▶ queue_request (计算 prefix 候选)
     │
     └─ server 立即调用 set_request_tag(id, tag)
            │  失败: 未知 id / 已 cache_reserved / block_table 非空
            ▼
     group.tag = T；linear 若已 active 则 set_seq_tag(T)
            │
         首次 admission: ensure_active(seq, T) / fork_checkpoint 保留 T
            │
         分配的每个 full 块 set_tag(T)
            │
         填满注册 → owned(T) 可见
            │
         fork_request: 子请求继承 T（子 slot 不再单独注入 tag）
            │
         finish / cancel: release_request → 块释放或保留为 cache-resident
```

要点：

- **分配 ≠ 可复用**：checkout 时就写 owner tag，但只有“填满 + 注册（full）”或
  “GPU checkpoint 创建（linear）”后才进入映射。
- **晚到无效**：请求一旦持有块（`cache_reserved || !block_table.empty()`），
  `set_request_tag` 返回 false，块上的 owner 不会被补写；服务端只记录 warning。
- **子请求**：`n_cmpl > 1` 的子 slot 由 `fork_request` 派生，tag 继承父请求；
  子任务的 `params.tag` 不单独生效。
- **未知/已完成 request_id**：id 在 `finish()` 一开始就从 `id_to_group` 擦除，
  晚到的 `set_request_tag` 返回 false。

## 5. 查询物化与计数口径（`llama_paged_tag_manager`）

`materialize(requests)` 一次完成：

1. `full->collect_tag_blocks(owned_full, &reusable_full)`：遍历 live prefix 项，
   tag 非空则加入 `owned_full[tag]`，同时把所有 live id 记入 `reusable_full`；
2. `linear->collect_tag_blocks(owned_linear)`：遍历 live GPU checkpoint；
3. 对每个带 tag 的活跃请求，把 `block_table ∩ reusable_full − owned_full[tag]`
   加入 `shared_full[tag]`。

派生查询：

| 查询 | 结果 |
| --- | --- |
| `usage(snapshot, tag)` | `full_blocks = owned_full[tag].size()`，`linear_blocks = owned_linear[tag].size()` |
| `all_usage` | 三个 map 的 tag 并集；只有 shared 的 tag 计数为 0 |
| `block_ids` | 三段 id 升序：full owned / full shared / linear owned |
| `for_each_block` | 每个 tag 先 owned 后 shared，full 后 linear；tag=nullptr 遍历全部 |
| `block_stats` | `total = get_num_gpu_blocks()`，`free = n_free_gpu_blocks()`，`used = total − free`；full/linear 独立，cpu 由 C API 填 0 |

不变式（e2e 测试逐条断言）：

- `len(full_owned) == full_blocks`，`len(linear_owned) == linear_blocks`；
- `sum(tag.full_blocks) <= npu.full.used_blocks`（linear 同理）；
- `owned` 与 `shared` 不相交；
- full 与 linear id 各自从 0 编号，不可混用。

## 6. 边界与容易误解的行为

| 场景 | 结果 / 原因 |
| --- | --- |
| partial tail block | 无 hash，不可复用 → 不计数 |
| linear ACTIVE state | 不可复用（copy-on-fork）→ 不打 tag、不计数 |
| 请求带 media | `register_tokens` 不注册 → full 块无 hash，不建映射 |
| prefix cache 关闭 | 不注册 → 无映射 |
| 只有 shared 的 tag | 计数 0，但 `blocks.full.shared` 非空 |
| owner 被重新注册顶替 | 命中方从 checkpoint 边界之后重算的块会用新物理块注册同一 hash 链，旧 cached 副本被释放 → **owner 计数可能下降**（实测 e2e 中 tenantA 由 6 变 4）。这是设计行为，不是泄漏 |
| swap_out / demote | 对统计而言等于直接释放 |
| swap_in | 对统计而言等于新分配一块无 tag 的块 |
| CPU 池 | 不计数、不打 tag |
| on_finish 回调 | 发生在 `finish()` 内、`release_request()` 之前，回调里查询会看到尚未释放的映射；文档要求不要在回调里查询 |
| tag 长度 | 服务端按 UTF-8 **字节**限制 128（中文约 42 字） |

## 7. 线程模型与并发

- 所有查询只允许在 scheduler / main loop 线程调用；`llama_paged_tag_manager`
  不加锁。
- 服务端 `SERVER_TASK_TYPE_KV_STATS` 在 `process_single_task()` 中处理，并被加入
  yield 白名单（设计文档要求）。当前 paged 路径唯一的 yield 区域是投机解码
  `port.yield([&]{ common_speculative_draft(port.spec_ctx()); })`，draft 在
  `ctx_dft` 上 decode，不修改目标 paged cache 的 block_manager / prefix_index，
  因此只读查询不会与写并发。**若以后新增会修改 paged cache 的 yield 区域，必须
  重新评估该白名单。**
- `materialize` 使用活跃请求的 `(tag, block_table)` 指针，生命周期限于本次调用。

## 8. 本次检查发现并加固的问题

1. **free block 被写入 tag**（`llama_block_manager::set_tag`）
   - 原实现：ref_count==0 时告警但仍写入。若调用方误用，该 tag 会残留到下一次
     `checkout_gpu_blocks()`（例如 `swap_in` 不经过 `allocate()` 补 tag），使
     swap 回来的块携带旧 owner。
   - 现实现：ref_count==0 时告警并**拒绝写入**。
2. **服务端两段式 id 查询的容量竞争**（`tag_blocks_to_json`）
   - 原实现：先取长度再一次性填充，忽略第二次调用的返回值；若两次调用之间
     快照变大，会返回全 0 / 截断的 id 列表。
   - 现实现：容量不足时按写回的长度扩容重试（最多 8 次），保证返回成功快照。
3. **yield 白名单的线程亲和性**：保留（设计文档要求），并在 §7 明确前提与
   复核条件。
4. **checkpoint fork 失败重试丢失 owner tag**（`llama_linear_state_cache::fork_checkpoint`）
   - 原实现：进入时从 `seq_state` 读取 tag 并立即 `release(seq_id)`；若随后
     allocate 触发了对被 fork checkpoint 的 demote/evict，重校验失败返回 false，
     此时 `seqs` 中已没有该请求。调用方
     `reserve_prefix_candidate()` 重匹配后**不再**调用 `ensure_active()`
     直接重试 `fork_checkpoint()`，于是重试路径拿到空 tag，新建的 ACTIVE
     state 与后续 checkpoint 都不带 owner tag → `linear_blocks` 少计。
   - 现实现：`fork_checkpoint(chain_hash, seq_id, tag)` 增加 tag 参数，
     `reserve_prefix_candidate()` 两处调用都传 `group.tag`；`ensure_active()`
     在已有 seq 且 tag 为空时也会补写，作为兜底。
5. **全量 `?blocks=1` 的重复物化**（`server_scheduler_paged::get_tag_usage`）
   - 原实现：列表里每个 tag 各调用一次 `get_tag_block_ids`，整体
     O(tags × (可复用块 + Σ block_table))。
   - 现实现：一次 `for_each_tag_block(NULL)` 物化，同时得到所有 tag 的 id 与
     计数（`full_blocks = len(owned)`、`linear_blocks = len(linear_owned)`），
     计数与 id 来自同一快照。
6. **非问题（复核确认）**：tag 字节长度限制、owner 重注册导致计数下降、
   shared 不计入计数、linear 只有 owned 无 shared、CPU 池恒 0。

## 9. 与设计文档的落点差异

- 设计文档把物化逻辑放在 scheduler；实现抽成独立类
  `llama_paged_tag_manager`：cache 只提供 `collect_tag_blocks()` 原语，scheduler
  只提供活跃请求视图，聚合/计数/id/遍历/池统计集中在 tag manager，便于单测与复用。
- 服务端 `server_task_result_tag_usage` 用预渲染的 `json payload` 承载响应体
  （语义与文档的 `tag/has_tag/tags/blocks` 字段一致），避免在 task 结构里重复
  维护两套字段。

## 10. 验证

服务化端到端测试位于 [`tests/tag_kvcache_manger/`](../../tests/tag_kvcache_manger/)，
与本文各节对应：

| 生命周期环节 | 用例 |
| --- | --- |
| 初始池统计 / 无映射 | `test_kv_stats_initial_consistency` |
| 分配→注册→结束驻留 | `test_tag_counts_owned_reusable_blocks` |
| GPU checkpoint 打 tag | `test_tag_linear_gpu_checkpoint` |
| 未知 tag | `test_unknown_tag_is_zero` |
| 列表/明细口径 | `test_tag_listing_and_blocks_flag` |
| 共享：只建映射不计数、命中方结束只删 shared | `test_shared_prefix_is_mapped_but_not_counted` |
| 请求入口（OAI / slots / 长度校验） | `test_oai_endpoints_accept_tag`、`test_slots_expose_request_tag`、`test_tag_longer_than_128_is_rejected` |
| 非 paged 服务 | `test_non_paged_server_reports_not_supported` |

```bash
TAG_TEST_MODEL=/home/qwen35-2b-f16.gguf \
python3 -m pytest -v tests/tag_kvcache_manger/test_tag_kv_server_e2e.py
```

## 11. 调试提示

- 请求结束后看不到块：先确认 prompt 是否跨过 checkpoint 晶格。hybrid 模型默认
  `--kv-prefix-ckpt-interval 4`、block=16，前缀复用至少需要 64 token；可用
  `--kv-prefix-ckpt-interval 1` 缩小下限。
- 区分“没有 tag”和“有 tag 但不可复用”：比较 `/tag_usage?tag=T`（计数 0）与
  `/kv_stats` 的 `used`（>0 说明块被占用但不可复用，如 partial tail、ACTIVE
  linear state、CPU/swap 块）。
- 观察共享：在命中方请求运行期间轮询 `/tag_usage?tag=<命中方>&blocks=1`，
  `blocks.full.shared` 出现即表示已建立共享映射；命中方结束后该列表清空，
  owner 的计数不受影响。
