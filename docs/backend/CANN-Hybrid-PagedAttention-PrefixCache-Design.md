# CANN Hybrid-Paged Prefix Cache 设计

状态: 设计稿 (2026-09-15)

可行性分析见 `CANN-Hybrid-PagedAttention-Design.md` 第 18 节。本文档
给出完整接入方案: 块粒度前缀共享 + GDN checkpoint slot, 全部改动收敛
在 paged 生态文件内, 不修改 ggml-cann kernel 与模型/graph 构建层。

## 1. 目标与范围

- 同 prompt / 共享 system prompt / 多轮对话场景下, 新请求复用已缓存
  前缀, TTFT 从全量 prefill 降为仅后缀 prefill。
- 支持两类 memory:
  - dense 全注意力 (Qwen3-8B PA 路径): 仅 KV 块共享。
  - hybrid (Qwen3.5 dense/MoE): KV 块共享 + GDN 状态分叉。
- 不支持: 跨进程/跨实例缓存, 非块对齐部分共享 (块内尾块不共享),
  KV 块内随机写。

## 2. 总体架构

```text
llama_paged_scheduler_impl (驱动层)
  - llama_paged_prefix_cache: 块 hash 链索引 + checkpoint 表
  - add_request: 前缀匹配 -> 块共享 -> 后缀 prefill 计划
  - prefill/decode: hash 链维护, checkpoint 创建/驱逐
llama_kv_cache_paged (注意力侧)
  - share_blocks(dst_group, src_ids, k): ref_count++ 共享, 不拷数据
llama_memory_hybrid_paged (混合侧)
  - seq_cp 透传: attn 侧块共享 + recr 侧状态分叉
llama_memory_recurrent (GDN 侧, 复用现有机制)
  - seq_cp + cell COW + s_copy 图输入 => checkpoint slot 状态快照
```

核心原则: 只共享"整块", 被共享块天然不可变 (写入永远发生在已有
内容之后), 因此不需要 CoW 数据拷贝; GDN 状态快照完全复用上游
recurrent 的 seq_cp / cell COW / s_copy 机制。

## 3. 关键机制

### 3.1 块 hash 链索引

每条 sequence 维护已完成块的 hash 链:

```text
hash[0] = H(0, tokens[0 .. B))
hash[i] = H(hash[i-1], tokens[i*B .. (i+1)*B))
```

- token 来源: scheduler 的 `logical_seq` (prompt + 采样 token)。
- 某块被填满 (跨块边界) 时计算 hash 并登记:
  `registry[hash] = {block_id, seq_id, block_idx, last_used}`。
- 匹配时除 hash 相等外, 还校验该块 token 逐个相等 (防碰撞)。
- registry 惰性失效: 命中时检查 `block_manager` 引用计数 > 0,
  否则剔除该表项。

### 3.2 注意力 KV 块共享

- `llama_block_manager` 增加 `add_ref_blocks(ids)`: 对已分配块
  `ref_count++`, 不分配新数据。
- 新 group 的 `block_table` 前 k 项指向共享物理块, 之后正常追加。
- 不可变性论证: 写槽位只落在 group 自己 token 数之后; 共享块都是
  "满块", 任何 group 都不会再写它; 源 group 的尾块 (不满) 不参与
  共享。
- 释放: `free_blocks` 现有逻辑按块 `ref_count--`, 归零才回收,
  共享块自然存活。
- 换入换出: 块不可变, 两个 group 可分别持有同一逻辑块的 GPU/CPU
  副本, 内容一致; swap 时同步更新 registry 中该 hash 的物理块号。

### 3.3 GDN checkpoint slot (复用上游机制)

GDN 状态是滚动值, 只有块边界快照可复用。方案不新建快照池, 直接用
recurrent memory 的 seq_cp + cell COW:

```text
创建 (seq S 跨越块边界 B 的当步内, 下一步 prepare 之前):
  seq_cp(S -> ckpt_slot)            // ckpt 与 S 共享 state@B 的 tail cell
  S 推进下一个 ubatch 时, prepare() 检测 cell 共享, COW 分裂:
  ckpt_slot 保留 state@B, S 获得 new cell (图内 s_copy 拷贝)

分叉 (新请求 R 命中前缀 k 块, 找到 chain 末块 hash[k-1] 的 checkpoint):
  seq_cp(ckpt -> R)                 // R 与 ckpt 共享 state@B cell
  R 后缀 prefill 首个 ubatch 触发 COW, R 独立出 new cell
```

- checkpoint slot 占用专用 seq id: 约定 `[n_seq_max - R, n_seq_max)`
  为保留段, scheduler 不把该段分配给请求; `n_seq_max` 按
  `n_parallel + R` 配置。
- 每个 checkpoint 占一个 recurrent cell (每层状态一块), 数量 R 默认 4,
  可配; 驱逐策略 LRU, 驱逐时对 slot 执行 `seq_rm` 释放 cell。
- cell.pos 在创建点即为 B, seq_cp 复制 tail 指针后 pos 语义保持。

### 3.4 后缀 prefill 与 pos 记账

- group 新增 `n_tokens_cached = k * B`; prefill 只处理
  `logical_seq[n_tokens_cached .. n_prompt)`。
- batch.pos = `n_tokens_cached + i`; RoPE / GDN pos 全部由此推导。
- `write_slots` / `context_lens` 按 block_table (含共享块) 与
  `n_tokens_cached` 推导, 与现有 scheduler 逻辑一致, 只需把"已有
  token 数"的来源从 0 改为 `n_tokens_cached`。
- `n_decoded` / 采样逻辑不变。

### 3.5 命中条件与回退

hybrid 路径 v1 要求: 匹配块数 k >= 1 且 `hash[k-1]` 存在活跃
checkpoint。不满足则整体回退为 k=0 全量 prefill (正确性优先)。

v2 再做优雅降级: 无 checkpoint 时仍共享注意力块, GDN 从最近
checkpoint 边界重扫前缀尾部; 该模式需要 "GDN-only / 跳过 KV 写入"
的 ubatch 模式, 涉及 graph input 扩展, 暂不进入 v1。

dense 路径无 GDN 状态问题, 只要 k >= 1 即可共享。

### 3.6 checkpoint 策略 (v1)

- 每个 seq 保留最新一个 checkpoint: prefill 完成时在
  `floor(n_prompt / B) * B` 处创建 (n_prompt >= B 才创建)。
- 准入时若源 seq 恰好停在匹配边界, 顺带补建 checkpoint。
- 同 prompt 重复请求 (bench prompt pool) 与多轮对话 (前缀含历史
  生成 token 且对齐块边界) 可命中; 共享前缀中途分叉且无 checkpoint
  时 v1 回退全量 prefill。

## 4. 数据结构

```cpp
struct paged_prefix_entry {
    uint64_t    hash;        // 块 hash 链值
    uint32_t    block_id;    // 物理块
    int32_t     owner_seq;   // 拥有者请求 seq id
    uint32_t    block_idx;   // 在 owner 链中的序号
    int64_t     last_used;
};

struct paged_ckpt_entry {
    uint64_t    chain_hash;  // 覆盖到该边界的链末块 hash
    int32_t     ckpt_seq;    // 保留段 seq id
    llama_pos   pos;         // 边界位置 k*B
    int64_t     last_used;
};

class llama_paged_prefix_cache {
    std::unordered_map<uint64_t, paged_prefix_entry> blocks;
    std::unordered_map<uint64_t, paged_ckpt_entry>   ckpts;
    // match(tokens) -> k; register_block(...); bind_ckpt(...); evict_lru(...)
};
```

## 5. 流程

### 5.1 准入 (add_request)

```text
1. tokenize prompt, 按 B 分块, 沿 registry 匹配最长前缀 k
   (逐块校验 hash + token, 物理块引用 > 0)
2. k > 0:
   a. share_blocks(group, matched_ids, k)          // attn 侧
   b. hybrid: 查 hash[k-1] 的 checkpoint;
      命中 -> seq_cp(ckpt -> R); 未命中 -> 回退 k=0
   c. group.n_tokens_cached = k*B; 预分配后缀块
3. prefill 只提交后缀 token, batch.pos 基准 = k*B
```

### 5.2 checkpoint 创建与失效

```text
prefill 完成步: 若 floor(n_prompt/B) >= 1, 取保留 slot (LRU),
  seq_rm(ckpt_seq), seq_cp(S -> ckpt_seq), 登记 chain_hash
源 seq 被 free / 回退越过边界: 对应 registry/ckpt 表项失效
```

### 5.3 释放与驱逐

```text
请求结束: free_blocks 按 ref-- ; 共享块由最后持有者回收
registry 惰性校验: 命中时检查块引用, 失效则剔除
checkpoint LRU: slot 不足时驱逐 last_used 最旧项
```

## 6. 改动清单 (文件级)

```text
新增  src/llama-paged-prefix-cache.h/.cpp
      hash 链 registry, checkpoint 表, 匹配/登记/驱逐
改    src/llama-paged-scheduler-impl.cpp/.h
      准入匹配, n_tokens_cached/pos_base, 链维护 (含解码追加),
      checkpoint 创建/驱逐, swap 时 registry 同步,
      seq id 保留段管理
改    src/llama-block-manager.h/.cpp
      add_ref_blocks(ids), 引用计数查询
改    src/llama-kv-cache-paged.h/.cpp
      share_blocks(), seq_cp/seq_keep 落地 (按 group 块表复制 + ref++)
改    src/llama-memory-hybrid-paged.h/.cpp
      seq_cp/seq_keep 去掉 throw: attn 块共享 + recr 透传
改    examples/paged/paged.cpp
      开关与统计 (命中块数, 节省 token 数, TTFT 对比)
不改  ggml-cann*, llama-model*, llama-graph*, llama-hybrid-paged-graph*
```

预估: dense 路径 (Phase 0/1) 约 400 行; hybrid checkpoint 接入
约 250-350 行; 降级重扫 (v2) 约 300 行。

## 7. 配置与内存预算

```text
LLAMA_PAGED_PREFIX_CACHE   默认 1
LLAMA_PAGED_PREFIX_CKPT    默认 4 (保留 checkpoint slot 数 R)
```

内存代价:

- KV 共享块: 0 (复用)。
- checkpoint: R 个 recurrent cell, 每 cell 约等于一条 seq 的 GDN
  状态 (Qwen3.5-9B 量级为几十 MB, 以 init 日志 memory_size_r/s 为准);
  R=4 时约 100-200 MB。
- `n_seq_max` 与 recurrent cell 池需覆盖 `n_parallel + R`。

## 8. 正确性风险与对策

1. hash 碰撞: 匹配时逐 token 校验。
2. checkpoint 时点差一: 必须在源 seq 处理完边界块最后 token 的当步
   内、下一次 prepare 之前创建; 单测覆盖。
3. COW 分裂顺序: 分叉后 R 的首个 ubatch 才分裂 cell; 期间 ckpt 与 R
   共享 cell 是安全的 (都未推进)。
4. 共享块换出: 块不可变, 双副本 (GPU/CPU) 内容一致; registry 在
   swap 后更新物理块号。
5. PA 元数据: block table 允许跨 seq 重复物理块号, PA 按 seq 独立
   gather, 无唯一性假设; 上板验证。
6. pos/n_decoded/上下文余量: 全部以 `n_tokens_cached + 后缀` 记账,
   分配余量按 block_table 现有块数推导 (现有公式天然兼容)。
7. 回退路径: 任何条件不满足即回退全量 prefill, 保证可回退性。

## 9. 测试计划

- 单测: hash 链与匹配; add_ref/free 引用计数; checkpoint COW 后
  state@B 与全新 prefill 的 state 逐元素一致。
- 精度: 双请求同 prompt, 第二请求输出与无缓存路径 argmax 一致;
  共享前缀下两 seq 并行 decode 输出一致。
- 功能: 缓存压力 (小 ngpub) 下驱逐正确; 共享块 swap in/out;
  n_prompt 非块对齐; prompt 短于一块 (不共享)。
- 性能: 同 prompt 二次请求 TTFT 对比 (期望接近后缀 prefill 时间);
  decode TPOT 无回归; CANN PA 上板跑通。

## 10. 分阶段实施

```text
Phase 0  基础设施: add_ref_blocks, prefix cache 骨架, 开关与统计
Phase 1  dense 命中: 块共享 + 后缀 prefill + pos 记账 (Qwen3-8B PA 验证)
Phase 2  hybrid 接入: checkpoint slot, seq_cp 透传, COW 分叉验证
         (Qwen3.5-2B 快速验证 -> 9B)
Phase 3  v2 降级: GDN 尾部重扫 + 周期性 checkpoint + LRU 调优
```

## 11. 验证记录 (2026-09-15, qwen3.5-2b F16)

场景: 同 prompt (2048/2112 token, 错峰到达), greedy, CANN PA。

```text
set_ckpt_slots: reserved 2 checkpoint slot(s) at seq ids [2, 4)
request 1 reuses 32 cached block(s) (2048 tokens), prefilling 64 suffix tokens
ttft min 115.7 ms (命中) / max 1994.9 ms (全量)
```

开/关缓存对比 (LLAMA_PAGED_PREFIX_CACHE=0/1): 两请求输出逐 token 一致,
GDN checkpoint 分叉 + 后缀 prefill 与全量重算等价 (argmax 级)。

补充实现说明: 请求结束后其已登记链保留为 cache-resident 块
(ref=1, LRU 驱逐), 使命中可跨请求存活; 已知限制: decode+prefill
混合 batch 的 metadata 断言为既有问题, 依赖错峰到达规避。
