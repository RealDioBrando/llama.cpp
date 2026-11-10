# Tag KV Block 统计与查询接口

## 1. 目标

请求可携带可选 string `tag`，按 tag 查询其占用的 KV block：

- full-attention 与 linear-attention **分开计数**（块大小不同，不合并）；
- 查询返回具体的 **block id 列表**，区分 owner 与共享引用；
- 口径覆盖**活跃请求引用的可复用块**和**请求结束后驻留 prefix cache 的块**；
- 只统计**可被 prefix cache 复用**的块；
- CPU 池不参与（本轮 CPU 逻辑待重构，所有 CPU 相关路径按"释放"处理）。

不在本轮范围：跨 tag 租户隔离、CPU 池统计、swap 后 tag 追溯、router 跨子实例聚合。

## 2. 三个概念

| 概念 | 定义 | 存放位置 |
| --- | --- | --- |
| **owner tag** | 物理 GPU block 的首次分配者 tag；checkout 时写入，ref 归 0 时清除 | `physical_block.tag` |
| **可复用 block** | full：full prefix index 中 live 的项（有 hash、generation 匹配、ref>0、GPU 池）；linear：`checkpoint_index` 中 live 的 GPU checkpoint | 各自的索引 |
| **tag → block id 映射** | owned（owner 是该 tag）+ shared（被该 tag 的请求引用、但 owner 是别人） | **查询时物化**，不增量维护 |

```
allocate ──► [GPU block, owner tag=a] ──► register_tokens() 算 hash ──► 可复用
                                              │
release / evict (ref==0) ──► tag.clear() ◄────┘
```

**关键区别：分配 ≠ 可复用。** full 块在 checkout 时就有 owner tag，但要等填满、算出 hash 并注册进 prefix index 后才进入映射；linear ACTIVE state 块永远不进入映射。

### 查询时物化

每次查询在 scheduler 线程一次性汇总，不维护增量状态：

1. **owned**：遍历 full `prefix_index.entries()` / linear `checkpoint_index.entries()`，筛出 live 且 GPU 的块，按 `block_manager.tag(id)` 归入 `owned_full[tag]` / `owned_linear[tag]`（集合去重）。
2. **shared**：遍历活跃请求，若 `group.tag` 非空，把 `block_table` 中属于 full 可复用集合、且不在 `owned_full[group.tag]` 里的 id 加入 `shared_full[group.tag]`。
3. **计数 = owned 集合大小；映射 = owned ∪ shared。**
4. 只有 shared、没有 owned 的 tag 也会出现在结果里（计数 0，映射非空）。

这样物理 block 上只维护一份 owner tag，共享引用完全由活跃请求的 `block_table` 推导，避免增量 refcount 的漏删/误删。

## 3. 记录与删除时机（核心）

### 3.1 记录（映射可见）

| 事件 | 代码路径 | 映射变化 |
| --- | --- | --- |
| full 分配 | `llama_kv_cache_paged::allocate()` checkout 后 `set_tag(id, group.tag)` | owner tag 写入，**尚未**进映射（无 hash） |
| full 块填满并注册 | `llama_paged_cache_manager::register_tokens()` → `register_prefix_block()` | `owned_full[group.tag]` **出现**该 id |
| prefix 命中复用 | `reserve_prefix_candidate()` → `add_ref_blocks` + `take_cached_ref`，写 `group.block_table` | 命中方的 `shared_full[tag]` **出现**该 id |
| linear checkpoint 创建（GPU） | `create_checkpoint()`：`set_tag(ckpt_id, seq_state.tag)` + `checkpoint_index.insert()` | `owned_linear[tag]` **出现**该 id |

### 3.2 删除（映射消失）

| 事件 | 代码路径 | 映射变化 |
| --- | --- | --- |
| 请求结束、块未保留 | `release_sequence()` → `free_blocks()` → `release_gpu_blocks()`，ref 归 0 | `tag.clear()`，owned **消失** |
| cancel / truncate / 直接释放 | 同上 | 同上 |
| prefix 淘汰 | `evict_cached_prefix_block()` → `release_block()` | 同上 |
| **命中方**请求结束 | `free_blocks()` 清空 `block_table` / `release_sequence()` | 只删该 tag 的 shared 条目；owner 映射不受影响 |
| linear checkpoint 淘汰 | `evict_checkpoint()`（remove + `release_block`） | `tag.clear()`，owned_linear **消失** |
| linear checkpoint 降级到 CPU | `demote_checkpoint()` → GPU block `release_block()` | GPU 侧映射**消失**；CPU checkpoint 不建映射 |
| full / linear swap_out | `swap_out()` → `free_blocks()` / `release_block()` 释放 GPU block（等价释放） | **消失**；索引转向 CPU block 后被 `is_gpu==false` 过滤 |
| full / linear swap_in | 换回的新 GPU block **不补 tag** | **不建立**（swapped 请求统计为 0） |
| `clear()` | 重建 block_manager + 清空索引 | 全部**消失** |

此外，索引项失效（generation 不匹配、被 swap 到 CPU、`invalidate_block`）也会让映射消失——查询的 live 过滤会保证这一点。

### 3.3 两个"独立"

- **owner 映射与 shared 映射独立**：命中方请求结束只删自己的 shared；只要原 owner 的块仍驻留，owner 映射继续存在。
- **owner tag 的写入/清除 ≠ 映射的可见/消失**：tag 在 checkout 写入、ref 归 0 清除；映射还要求"块可被 prefix cache 复用"。

## 4. 边界与特例

| 场景 | 结果 |
| --- | --- |
| 空 tag 请求 | 分配的块不归属任何 tag（仍是全局 `used` 的一部分） |
| `set_request_tag` 晚于首次调度 | 返回 false；块上的 owner tag 不补写 → 该请求不产生映射 |
| prefix cache 关闭 / 请求带 media | `register_tokens()` 不注册 → full 块永远无 hash → 不建映射 |
| linear ACTIVE state 块 | 不可复用（copy-on-fork）→ 不打 tag、不建映射 |
| linear checkpoint 命中 | `fork_checkpoint()` 拷贝到新 ACTIVE 块 → checkpoint 本身不共享，新块不建映射 |
| CPU 池 block（任何来源） | 不打 tag、不建映射、不计数 |
| swap_out / demote_checkpoint | 对统计而言 = **直接释放** |
| swap_in | 对统计而言 = **新分配一块无 tag 的 block** |

## 5. 计数与输出口径

- `npu.full`、`npu.linear` 各自独立，`used = total - free`；**不提供两类相加的合并块数**。
- `cpu.full` / `cpu.linear` 恒为 0（预留，等 CPU 池重构）。
- tag usage：
  - `full_blocks` = owner 为该 tag 的 live 可复用 full block 数；
  - `linear_blocks` = owner 为该 tag 的 live GPU checkpoint 数；
  - 共享块只归 owner，因此 `sum(tag.full_blocks) <= npu.full.used`、`sum(tag.linear_blocks) <= npu.linear.used`。
- block id 列表：`owned` 长度严格等于上述计数；`shared` 只出现在映射里，不计数；id 升序返回。
- **full 与 linear 的 id 各自从 0 编号**，必须分两个集合、API 用 `kind` 区分，禁止混用。
- `on_finish` 回调发生在 `finish()` 内、`release_request()` 之前，**不要在回调里查询**。

## 6. 内部接口

### 6.1 block_manager（src/llama-block-manager.h/.cpp）

- `physical_block` 增加 `std::string tag;`（owner tag）。
- 新增 `set_tag(uint32_t id, const std::string & tag)`（`ref_count==0` 时告警）与 `const std::string & tag(uint32_t id) const`（越界返回空串）。
- 唯一清除点：`release_gpu_blocks` / `release_cpu_blocks` 中 ref 归 0 时 `block.tag.clear()`。
- 不做 `collect_tag_usage`：统计遍历的是 prefix/checkpoint 索引，不是 registry。

### 6.2 full-attention（src/llama-kv-cache-paged.h/.cpp）

- `allocate()`：checkout 后对每个新 block `set_tag(id, group.tag)`。
- `swap_out/swap_in`：不打 tag、不搬 tag（swap_out 等价释放；swap_in 不补 tag）。
- `add_ref_blocks` / `share_blocks` / `copy_block_content` / `register_prefix_block` / `release_sequence`：不改 tag。
- `llama_full_prefix_index` 增加 `entries()` 访问器（对齐 linear 索引）。
- 新增：

```cpp
void collect_tag_blocks(std::unordered_map<std::string, std::unordered_set<uint32_t>> & owned,
                        std::unordered_set<uint32_t> * reusable_ids = nullptr) const;
```

遍历 `prefix_index.entries()`，只保留 `is_gpu(id)`、`ref_count > 0`、`generation == e.generation` 的项；`tag(id)` 非空则加入 `owned[tag]`；同时把 id 加入 `reusable_ids`。

### 6.3 linear-attention（src/llama-linear-state-cache.h/.cpp）

- `seq_state` 增加 `std::string tag;`；新增 `set_seq_tag(seq_id, tag)`。
- `ensure_active(seq_id, tag)`：只写 `seq_state.tag`，**不给 ACTIVE block 打 tag**。
- `seq_cp` / `fork_checkpoint`：继承/设置 `seq_state.tag`；新 ACTIVE block 不打 tag。
- `create_checkpoint()`：checkpoint 落在 GPU 池时 `set_tag(ckpt_id, seqs[seq_id].tag)`；落 CPU 池不 tag。
- `demote_checkpoint` / `evict_checkpoint` / `swap_out/swap_in`：不搬 tag（等价释放/不补 tag）。
- 新增 `get_num_gpu_blocks()` / `get_num_cpu_blocks()` 与：

```cpp
void collect_tag_blocks(std::unordered_map<std::string, std::unordered_set<uint32_t>> & owned) const;
```

遍历 `checkpoint_index.entries()`，用 `checkpoint_is_live(e)` 且 `is_gpu(e.block_id)` 过滤，按 `tag(e.block_id)` 归集。

### 6.4 scheduler（src/llama-paged-scheduler-impl.h/.cpp）

- `llama_sequence_group` 增加 `std::string tag;`；`fork_request()` 子请求继承。
- `reserve_prefix_candidate()` 三处 `ensure_active` 传 `group.tag`。
- 新增：
  - `bool set_request_tag(int32_t request_id, const std::string & tag)`：若 `cache_reserved || !block_table.empty()` 返回 false；否则写 `group.tag`（若 linear 已 active 调 `set_seq_tag`）。
  - `materialize_tag_blocks()`：按第 2 节的规则一次性物化 `owned_full / owned_linear / shared_full`，供下面所有查询共用。
  - `all_tag_block_usage()` / `tag_block_usage(tag)` / `tag_blocks(tag)` / `for_each_tag_block(...)`。
  - `kv_cache_block_stats()`：`npu.full` 取 full `get_num_gpu_blocks()/n_free_gpu_blocks()`，`npu.linear` 取 linear 同名接口，`cpu.*` 全 0。
- `fork_request()` 在 `seq_cp` 后 `set_seq_tag(dst_id, child.tag)`。
- 查询只在 scheduler 线程调用，不加锁。

## 7. 公共 C API（include/llama.h）

```c
struct llama_paged_kv_cache_block_stats {
    uint64_t total_blocks;
    uint64_t used_blocks;
    uint64_t free_blocks;
};

struct llama_paged_kv_cache_pool_stats {
    struct llama_paged_kv_cache_block_stats full;
    struct llama_paged_kv_cache_block_stats linear;
};

struct llama_paged_kv_cache_stats {
    struct llama_paged_kv_cache_pool_stats npu;
    struct llama_paged_kv_cache_pool_stats cpu;  /* reserved, all zero */
};

struct llama_paged_tag_pool_usage {
    uint64_t full_blocks;    /* owner 的可复用 full block 数 */
    uint64_t linear_blocks;  /* owner 的 GPU checkpoint 数 */
};

struct llama_paged_tag_block_usage {
    struct llama_paged_tag_pool_usage npu;
    struct llama_paged_tag_pool_usage cpu;  /* reserved, all zero */
};

LLAMA_API bool llama_paged_scheduler_set_request_tag(struct llama_paged_scheduler * sched,
                                                     int32_t request_id, const char * tag);

LLAMA_API bool llama_paged_scheduler_get_tag_block_usage(struct llama_paged_scheduler * sched,
                                                         const char * tag,
                                                         struct llama_paged_tag_block_usage * out);

typedef void (*llama_paged_tag_usage_cb)(const char * tag,
                                         const struct llama_paged_tag_block_usage * usage,
                                         void * user_data);
LLAMA_API void llama_paged_scheduler_for_each_tag_block_usage(struct llama_paged_scheduler * sched,
                                                              llama_paged_tag_usage_cb cb,
                                                              void * user_data);

/* kind: 0 = full, 1 = linear; owned: 1 = 计入计数，0 = 仅共享引用
 * tag 为 NULL 时遍历所有 tag */
typedef void (*llama_paged_tag_block_cb)(const char * tag, int32_t kind, uint32_t block_id,
                                         int32_t owned, void * user_data);
LLAMA_API bool llama_paged_scheduler_for_each_tag_block(struct llama_paged_scheduler * sched,
                                                        const char * tag,
                                                        llama_paged_tag_block_cb cb,
                                                        void * user_data);

/* 单 tag 的 block id 明细，两段式：
 *   1) 各 out 传 NULL：把所需长度写入 *n_*；
 *   2) 传入数组、*n_* 传容量：足够则填充并返回 true；不足则写回所需长度并返回 false。 */
LLAMA_API bool llama_paged_scheduler_get_tag_block_ids(
    struct llama_paged_scheduler * sched,
    const char * tag,
    uint32_t * full_owned_out,   size_t * n_full_owned,
    uint32_t * full_shared_out,  size_t * n_full_shared,
    uint32_t * linear_owned_out, size_t * n_linear_owned);

LLAMA_API bool llama_paged_scheduler_get_kv_cache_stats(struct llama_paged_scheduler * sched,
                                                        struct llama_paged_kv_cache_stats * out);
```

语义要点：

- 计数只算 owner；shared 不计入。
- `get_tag_block_ids` 的 id 按升序；`full_owned` 长度 == `full_blocks`，`linear_owned` 长度 == `linear_blocks`。
- 未知 tag：usage 全 0、id 长度为 0，均返回 true。
- 只有 shared 的 tag：计数 0，但 `full_shared` 非空。
- 所有查询必须在 scheduler/main loop 线程调用。

## 8. HTTP API（tools/server）

### 8.1 接收 tag

- `task_params` 增加 `std::string tag;`，`make_llama_cmpl_schema()` 增加 `tag` 字段（≤128 字符），覆盖 `/completion`、`/v1/completions`、`/v1/chat/completions`、`/v1/responses`、`/v1/messages`。
- `task_params::to_json()` 在非空时输出 `tag`，`/slots` 可见。
- `server_scheduler_paged::launch` 的文本路径与 multimodal 路径在 `add_request*` 成功后立即 `llama_paged_scheduler_set_request_tag(...)`。

### 8.2 查询端点

- 新增 `SERVER_TASK_TYPE_KV_STATS`，结果类型：
  - `server_task_result_kv_stats { llama_paged_kv_cache_stats stats; }`
  - `server_task_result_tag_usage { std::string tag; bool has_tag; std::unordered_map<std::string, llama_paged_tag_block_usage> tags; json blocks; }`
- `server_scheduler_interface` 增加 `get_kv_stats()` / `get_tag_usage()`；impl 转发上面的 C API。
- `process_single_task()` 加 case；yield allow-list 加新 task type；`init_routes()` 仿 `get_slots`；`server.cpp` 注册：
  `GET /kv_stats`、`GET /tag_usage`。
- 单 tag 查询默认带 `blocks`；全量列表默认只给计数，`?blocks=1` 才带 id。

```json
GET /kv_stats
{"npu":{"full":{"total_blocks":N,"used_blocks":U,"free_blocks":F},
        "linear":{"total_blocks":n,"used_blocks":u,"free_blocks":f}},
 "cpu":{"full":{"total_blocks":0,"used_blocks":0,"free_blocks":0},
        "linear":{"total_blocks":0,"used_blocks":0,"free_blocks":0}}}

GET /tag_usage
{"tags":{"tenantA":{"npu":{"full_blocks":x,"linear_blocks":y},
                    "cpu":{"full_blocks":0,"linear_blocks":0}}}}

GET /tag_usage?tag=tenantA
{"tag":"tenantA","npu":{"full_blocks":x,"linear_blocks":y},
 "cpu":{"full_blocks":0,"linear_blocks":0},
 "blocks":{"full":{"owned":[3,4,5],"shared":[9]},
           "linear":{"owned":[1],"shared":[]}}}
```

## 9. 测试

### 单元测试（tests/test-paged-kv.cpp）

1. `test_block_manager_tag_lifecycle`：checkout → set_tag → add_ref/release（保留）→ 最后一次 release（清除）；越界返回空串。
2. `test_scheduler_tag_full_reusable_only`：32-token prompt（block 16）→ `full_blocks == 2`；`full_owned` 正是这两个 id；多预留的 decode block 无 hash、不在列表里。
3. `test_scheduler_tag_resident_after_finish`：请求结束后 2 个 cached block 驻留，计数与 `full_owned` 不变。
4. `test_scheduler_tag_evict`：淘汰后计数下降/归零，`full_owned` 同步减少。
5. `test_scheduler_tag_two_tags_shared_prefix`：A 留下 2 块，B 命中 2 块 + 新分配 1 块 → `A.full_blocks == 2`、`B.full_blocks == 1`；B 的 `full_shared` 是 A 的 2 个 id，`full_owned` 是自己那 1 个；`sum(owned) == 可复用块总数`。
6. `test_scheduler_tag_linear_checkpoint`（hybrid）：GPU checkpoint → `linear_blocks == 1`；ACTIVE state 不计入；`demote_one_checkpoint()` 后为 0。
7. `test_scheduler_tag_no_hash_not_counted`：prompt 短于 block_size 或带 media → `full_blocks == 0`。
8. `test_kv_cache_stats_split`：`npu.full` / `npu.linear` 各自 `used + free == total`；`cpu.*` 全 0；无合并字段。
9. `test_set_request_tag_after_admission`：已分配 block 后 set 返回 false，计数不变。
10. `test_tag_cpu_paths_equate_release`（fixture 支持 swap 时）：`swap_out` 后为 0；`swap_in` 后仍为 0；`demote_checkpoint` 后 linear 为 0。

### e2e / server

- `tests/test-paged-kv-e2e.cpp`：`test_prefix_reuse` 设 tag；结束后断言 `npu.full` / `npu.linear` 自洽、`full_blocks > 0`、`full_owned` 长度一致；hybrid 下覆盖 `linear_blocks >= 1`。
- `tools/server/tests/unit/test_paged_server.py`：`/completion` 带 tag；`/tag_usage?tag=t1` 计数与 `blocks.full.owned` 一致；`/kv_stats` 自洽；驻留后仍 >0；第二个请求命中后 `blocks.full.shared` 非空但不计数；`/tag_usage?blocks=1` 全量带 id；OAI 接口 tag 生效；非 paged 返回 not supported。

### 本地验证

```bash
cd /home/h00910036/llama_schedule/llama.cpp
cmake --build build -j --target test-paged-kv llama-server test-paged-kv-e2e
./build/bin/test-paged-kv
PAGED_TEST_MODEL=/path/to/model.gguf ./build/bin/test-paged-kv-e2e -m /home/qwen35-2b-f16.gguf
./build/bin/llama-server -m /path/to/model.gguf --kv-paged -ngl 99 -b 2048 -ub 2048 -np 4
curl -s localhost:8080/kv_stats
curl -s 'localhost:8080/tag_usage?tag=t1'     # 含 blocks.full.owned/shared
```

## 10. 实现落点速查

| 文件 | 改动 |
| --- | --- |
| `src/llama-block-manager.h/.cpp` | `physical_block::tag`、`set_tag/tag`、ref 归 0 清除 |
| `src/llama-kv-cache-paged.h/.cpp` | 分配时打 tag；`collect_tag_blocks`；`entries()`；CPU 数量 getter |
| `src/llama-linear-state-cache.h/.cpp` | `seq_state.tag`、`ensure_active(tag)`、checkpoint 打 tag、`collect_tag_blocks` |
| `src/llama-sequence-group.h` | `std::string tag` |
| `src/llama-paged-cache-manager.cpp` | `ensure_active(seq_id, group.tag)` |
| `src/llama-paged-scheduler-impl.h/.cpp` | `set_request_tag`、物化 helper、4 个查询入口 |
| `include/llama.h` + `src/llama-paged-scheduler.cpp` | 新增 C 结构与函数 |
| `tools/server/*` | `tag` 字段、KV_STATS task、`/kv_stats`、`/tag_usage` |

## 11. 风险与已知限制

- **统计的是"可复用 KV 足迹"，不是进程物理占用**：partial tail block、linear ACTIVE state、swap/CPU block 不在统计里，因此 tag 计数可能低于请求实际占用的物理块数。
- **共享只建映射不计数**：`sum(tag counts) <= used`；共享 id 只在命中方的 `shared` 列表里可见。若以后要按持有者计费，加一个 referenced 计数即可（映射已物化）。
- **full/linear id 独立编号**：两者都从 0 开始，必须用 `kind` 区分。
- **无 tag 的 owner**：空 tag 的块不归属任何 tag；被有 tag 的请求共享时只出现在对方的 `shared` 里。
- **CPU 相关按释放处理**：swap_out / demote / 直接落 CPU 都视为释放；swap_in 不补 tag。CPU 池重构时需要重新定义换入时的归属与是否追溯。
- **查询开销**：每次 O(可复用块数 + Σ block_table)，stats 端点频率低可接受；若要高频轮询再改增量维护。
- **线程安全**：只允许 scheduler/main loop 线程调用；不要在 `on_finish` 回调里查询。
- **watermark**：`free_blocks` 返回原始空闲数（不减 watermark）。
- **重复 request_id**：请求完成后 id 被擦除，晚到的 `set_request_tag` 返回 false。
