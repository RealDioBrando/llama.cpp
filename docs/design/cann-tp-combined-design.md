# llama.cpp CANN TP 综合设计与当前实现

> 本文合并 `cann-tp-design.md` 的通用 TP 通信设计与 v4 310P-first 多线程重构方案，并补充当前 `cann-tp` 分支的实际实现状态。

## 1. 目标与边界

llama.cpp 的 CANN TP 目标是把多张 Ascend NPU 当作一个逻辑设备，支持 `--split-mode tensor`。当前实现优先面向单机 2 x 310P，并在以下能力上取得进展：

1. Meta Backend 通用 TP 切分与通信调度。
2. CANN 后端 per-device worker 和 HCCL AllReduce。
3. Paged attention + TP2。
4. Qwen3.5 / Qwen3.5 MoE 的本地 TP 推理。

当前不做：

1. 多机 TP。
2. MoE Expert Parallel。
3. 通用 ACL Graph capture/replay。
4. 4/8 卡的系统性验证。
5. FP16 之外的主流 TP 路径。

## 2. 总体架构

```text
llama model split policy
        |
        v
GGML Meta Backend
  - split state 推导
  - local tensor 分片
  - local graph 构造
  - PARTIAL 收集
  - collective 调度
        |
        v
CANN backend x N
  - per-device worker
  - device / context / stream
  - local graph compute
  - HCCL AllReduce
        |
        v
ACLNN / ACL Runtime / HCCL
```

核心原则：

1. TP policy 在 llama.cpp / GGML 上层，不在 CANN kernel 内重复推导。
2. Meta Backend 负责切分、局部图、PARTIAL 传播和 collective。
3. CANN backend 只执行已经切好的 local tensor。
4. 多设备时每个 CANN backend/context 由一个专属 worker 独占。
5. HCCL AllReduce 使用 in-place 语义，通信后不做每步 host synchronize。

## 3. 当前实现状态

| 能力 | 状态 | 说明 |
|---|---|---|
| Meta TP | 已实现 | split state、local graph、PARTIAL/collective |
| CANN per-device worker | 已实现 | 多设备强制启用，可环境变量关闭用于诊断 |
| HCCL AllReduce | 已实现 | F16/F32/BF16，in-place，设备 stream 有序 |
| Dense TP | 已跑通 | 主要以 2 卡验证 |
| Paged attention TP2 | 已跑通 | 9B / 35B smoke |
| MoE TP | 已跑通 | 使用 grouped matmul，非 Expert Parallel |
| GDN | 部分优化 | 使用 vendor fused path，性能未收敛 |
| ACL Graph | TP 模式关闭 | capture/replay 后置 |
| 4/8 卡 | 未系统验证 | 通信和拓扑待做 |
| 多机 | 未实现 | 不在本阶段 |

当前远端 `cann-tp` 分支包含部分未提交的 paged attention / Meta backend 修复。本文以当前实际可运行版本为准。

## 4. 模型切分与 Meta Backend

llama.cpp 使用通用 Meta Device 聚合多个 simple backend。模型加载时，`src/llama-model.cpp` 中的 `llama_meta_device_get_split_state()` 决定 tensor 如何切分：

1. Q/K/V/FFN 按对应维度切分。
2. KV cache 按 head 维度切分。
3. 大部分 bias、norm、router、meta 输入保持 mirrored。
4. `PARTIAL` 表示每卡有局部和，需要在层尾 AllReduce。

Meta Backend 不依赖具体硬件后端。它为每个 simple backend 生成 local graph，识别 PARTIAL tensor，然后调用后端通信 hook 或使用通用 butterfly fallback。

关键实现点：

1. `ggml-backend-meta.cpp` 负责 split state 推导、local graph 构造、AllReduce 调度。
2. `ggml-backend-meta.cpp` 中新增了 `GGML_OP_PAGED_ATTN` 的 split state 规则。
3. `ggml-backend-meta.cpp` 会探测 `ggml_backend_comm_init`、`ggml_backend_comm_free`、`ggml_backend_comm_allreduce_tensor`。
4. CANN backend 已经注册了这套通信 hook。

## 5. CANN per-device worker

多设备不能简单用同一个线程来回切 device。当前 CANN runtime 的 OpExecCache 是线程级缓存，跨设备同线程执行容易命中错误缓存。因此当前实现引入 per-device persistent worker。

设计要点：

1. 每个 CANN backend/context 独占一个 worker thread。
2. worker 启动时绑定自己的 device。
3. local graph compute 由 worker 执行。
4. collective 也在对应 device 的 worker 中提交。
5. worker 返回的是 host-side submission 完成，不是 device completion。
6. 后续 collective 和 graph op 依赖 device stream ordering，不额外 host synchronize。
7. 多设备默认启用 worker，`GGML_CANN_DISABLE_WORKER=1` 用于诊断。

worker 主要代码位于：

1. `ggml/src/ggml-cann/ggml-cann.cpp`
2. `ggml/src/ggml-cann/cann-worker.h`

## 6. HCCL 通信设计

当前 CANN TP 使用 HCCL 作为主通信路径。

### 6.1 communicator

`HcclCommInitAll()` 根据参与 TP 的 device list 创建每个 rank 的 communicator。当前实现是单机单进程、device rank 与 backend 顺序一一对应。

### 6.2 AllReduce

`ggml_backend_cann_comm_allreduce_tensor()` 的行为：

1. 检查所有 rank 的 dtype、element count 和 contiguity。
2. 将 GGML dtype 映射到 HCCL dtype：F16/F32/BF16。
3. inactive shard 先清零。
4. 每个 rank 在自己的 device worker 中调用 `HcclAllReduce()`。
5. 使用 in-place AllReduce。
6. 不做 host synchronize，依赖 device stream ordering。

### 6.3 回退路径

Meta Backend 自带 butterfly fallback。如果 HCCL hook 不存在或失败，仍然可以退回通用路径。旧的 D2D Copy + Add 设计保留为后续 fallback 方向，但当前主路径是 HCCL。

## 7. Paged attention + TP

### 7.1 语义

`GGML_OP_PAGED_ATTN` 负责一个 full attention 层的：

1. 新 K/V 写入 paged pool。
2. decode 或 prefill attention。
3. 输出 local head slice。

Qwen3.5 这类 hybrid 模型只有部分层使用 full attention。GDN/linear attention 层仍然走独立 recurrent 路径，不进入 `PAGED_ATTN`。

### 7.2 Meta split 规则

当前规则是：

1. Q/K/V/output 按 head 相关维度切分，即 `AXIS_1`。
2. K/V pool 按 `AXIS_2` 切分，对应 `n_heads_kv`。
3. block table、context lens、batch offsets、batch lens 等 metadata 全部 mirrored。
4. 所有输入都 mirrored 时，输出也 mirrored。

K/V pool 名称固定为：

```text
cache_paged_k_l%d
cache_paged_v_l%d
```

这样 Meta Backend 能稳定识别 paged KV cache。

### 7.3 CANN ATB 路径

`paged_attn.cpp` 使用 ATB：

1. `ReshapeAndCache` 写入新 K/V。
2. decode 使用 `PagedAttention`。
3. 无历史上下文的 prefill 使用 `SelfAttention`。
4. 带历史上下文的 prefill 使用 SplitFuse。
5. 每个设备维护独立 ATB context / workspace / op cache。

TP2 下必须 per-device 保存 ATB state，否则两张卡会复用同一个 ATB context/stream/workspace，导致卡死或错误。

### 7.4 Paged KV cache

`llama-kv-cache-paged.cpp` 维护：

1. 每 layer 的 K/V pool。
2. block manager。
3. block table。
4. sequence position。
5. GPU/CPU block swap。

当前 `seq_rm()` 支持整段 clear 和尾部截断，不支持中间区间删除。prefix cache 的状态 fork 只作用于 GDN state，paged KV 的 block sharing 由 paged scheduler 驱动。

## 8. MoE TP

当前 MoE 使用 TP weight slicing，不是 Expert Parallel。

### 8.1 切分方式

1. `ffn_gate_exps` / `ffn_up_exps` 按 `AXIS_1` 切分。
2. `ffn_down_exps` 按 `AXIS_0` 切分。
3. router / bias / norm 等保持 mirrored。

这等价于把每个专家的 FFN 输出和输入维度分到不同 device 上，最后由 Meta Backend 做 partial reduce。

### 8.2 CANN 路径

当前 MoE grouped path 使用：

1. `MoeInitRoutingV2` 完成 token-to-expert routing。
2. `GroupedMatmulWeightNz` 执行 NZ expert weight matmul。
3. 不满足 NZ 条件时退到 `GroupedMatmulV4`。
4. F16 expert weight 在加载后转成 NZ。

相关代码：

1. `ggml/src/ggml-cann/moe_ops.cpp`
2. `ggml/src/ggml-cann/moe_fused_ops.cpp`
3. `ggml/src/ggml-cann/weight_layout.cpp`

### 8.3 当前限制

1. 没有做 Expert Parallel。
2. 没有 token-level allgather dispatch。
3. 35B-A3B 的性能瓶颈主要在 MoE / GDN / TP 通信，不在 paged attention 本身。
4. vLLM Ascend 310P 使用显式 allgather + expert 分发，与当前 llama.cpp 的 TP weight slicing 不是同一种并行结构。

## 9. GDN / linear attention

Qwen3.5 是 hybrid 模型。GDN 层使用独立 recurrent/chunk path：

1. `LLM_FUSED_OP_GDN_AR`
2. `LLM_FUSED_OP_GDN_CH`

当前实现 port 了 vendor fused GDN op，但 TP 下的 shape/communication 优化还没收敛。因此 GDN 是当前 decode 性能的重要瓶颈之一。

## 10. Graph 与 stream

### 10.1 ACL Graph

TP 模式下当前关闭 ACL Graph。原因：

1. 多设备 capture 语义复杂。
2. worker 与 graph capture 生命周期需要更严格的绑定。
3. paged attention 与 metadata 更新不宜直接进入 capture/replay。

后续目标是串行 capture + 并行 replay，并且把 paged attention 的 graph params 显式更新。

### 10.2 prefill/decode 区分

当前通过 `ggml_backend_cann_set_prefill()` 显式把 prefill/decode phase 传给 CANN backend，避免从 graph shape 猜测执行阶段。

### 10.3 同步语义

HCCL AllReduce 是 stream-ordered。collective 后不做 host synchronize。后续 local graph op 也提交到同一 device stream，因此顺序依赖由 stream 保证。

## 11. 验证数据

当前 2 x 310P 的主要验证数据：

| 模型 | 配置 | 结果 |
|---|---|---|
| Qwen3.5 9B F16 | paged + TP2, p128/n128 | decode aggregate 6.22 tps, TPOT 163.2 ms |
| Qwen3.5 35B-A3B F16 | paged smoke, p32/n8 | decode aggregate 3.44 tps, TPOT 379.8 ms |
| Qwen3.5 35B-A3B F16 | non-paged baseline | pp32 13.88 tps, tg1 3.85 tps |

结论：

1. Paged attention + TP2 功能链路已经跑通。
2. 35B 的 paged 输出乱码与非 paged 对照一致，说明不是 paged attention 单独引入的精度问题。
3. 性能瓶颈主要在 MoE、GDN、TP 通信和 graph/调度层，不是 paged attention kernel 本身。

## 12. 实施路线

### Phase 1: 稳定 2 x 310P TP

1. 整理并提交当前 paged attention 修复。
2. 固化 TP2 dense / MoE / paged smoke。
3. 建立单卡 vs TP2 logits 对照。
4. 明确最小 correctness gate。

### Phase 2: paged attention 优化

1. block size 从 16 对齐到 64/128。
2. 减少 metadata 拷贝。
3. 优化 Q/K/V cast/permute。
4. 验证 ATB workspace 复用。
5. 引入 profiling 时间线。

### Phase 3: MoE 性能

1. 对比 `GroupedMatmulWeightNz` 与 `npu_grouped_matmul`。
2. 评估 allgather dispatcher。
3. 评估 Expert Parallel。
4. 优化 router / sort / gather / cast。

### Phase 4: GDN 性能

1. 确认 GDN 在 TP local shape 下的最优 kernel。
2. 减少 GDN 输入输出 cast。
3. 对齐 vLLM 310P 的 recurrent/chunk 路径。

### Phase 5: 4/8 卡

1. HCCL communicator 泛化。
2. 非均匀 split 验证。
3. topology-aware benchmark。
4. cross-device transport 优化。

### Phase 6: Graph 与平台扩展

1. ACL Graph 串行 capture。
2. 并行 replay。
3. BF16 capability gate。
4. 910B / A2 验证。

## 13. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| OpExecCache 线程级缓存 | 多设备失败 | 保持 per-device worker |
| 同一 CANN context 被多线程访问 | race | context/backend/stream 独占 |
| HCCL communicator 初始化差异 | 兼容性 | capability gate + fallback |
| cross-device copy 缺失 | transport fallback 受限 | 先保留 Host staging 方案 |
| local shape 不满足 kernel | TP 正确性 | FP16 dense 优先，逐层 dump |
| MoE TP 不等于 EP | 性能差距 | 后续评估 allgather / EP |
| paged block size 16 | PA kernel 效率低 | 对齐 64/128 |
| ACL Graph capture | TP 正确性 | TP 模式关闭，后置 |
| 非 host sync 后的依赖错误 | hang / 错值 | 依赖 stream ordering，禁止乱序提交 |

## 14. 与 vLLM Ascend 310P 的差异

| 维度 | vLLM Ascend 310P | 当前 llama.cpp |
|---|---|---|
| Paged attention | `torch_npu._npu_paged_attention` / splitfuse | ATB `PagedAttention` / `SelfAttention` |
| KV layout | 合并 5D cache | 独立 K/V pool，NZ 视图 |
| TP | worker/rank，显式 parallel group | Meta Backend 单进程多 backend |
| MoE | allgather dispatcher + expert 分发 | TP weight slicing + grouped matmul |
| GDN | 310P 专用 recurrent/chunk kernel | vendor fused path，仍在收敛 |
| Graph | 更完整的 ACL graph / compile pass | TP 模式关闭 |

因此 vLLM 310P 不能作为 1:1 参考实现，但可以作为 block size、mask、MoE dispatch 和 graph pass 的参考方向。

## 15. 文件索引

| 文件 | 作用 |
|---|---|
| `src/llama-model.cpp` | 模型 tensor split policy |
| `ggml/src/ggml-backend-meta.cpp` | Meta TP、local graph、collective |
| `ggml/src/ggml-cann/ggml-cann.cpp` | CANN backend、worker、op dispatch |
| `ggml/src/ggml-cann/cann-comm.cpp` | HCCL communicator / AllReduce |
| `ggml/src/ggml-cann/cann-worker.h` | worker 接口 |
| `ggml/src/ggml-cann/paged_attn.cpp` | ATB paged attention |
| `ggml/src/ggml-cann/moe_ops.cpp` | grouped MoE path |
| `ggml/src/ggml-cann/weight_layout.cpp` | dense/expert weight NZ layout |
| `src/llama-kv-cache-paged.cpp` | paged KV cache |
| `src/llama-memory-hybrid-paged.cpp` | paged attention + GDN hybrid memory |
| `src/llama-context.cpp` | graph phase / paged KV placement |

## 16. 总结

当前方案的核心是：

1. llama.cpp 继续持有 TP policy。
2. Meta Backend 继续负责切分和 collective。
3. CANN backend 提供 per-device worker、HCCL AllReduce、paged attention 和 grouped MoE。
4. 2 x 310P TP 已跑通 dense、MoE 和 paged attention。
5. 下一步重点不是重新设计 TP，而是优化 paged attention、MoE dispatch、GDN path 和 graph 层。
