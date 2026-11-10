# CANN PagedAttention 优化记录

状态：工程记录

更新时间：2026-11-10

本文汇总 CANN PagedAttention、混合 GDN、server 集成和 ACL graph 的优化过程。它是工作记录，不是设计方案。每条记录包含修改内容、实测收益和证据。

## 范围和约定

- 设备：除特别说明外均为 Ascend 310P3。
- 软件栈：CANN 9.1.0 正式版，ATB PagedAttention。
- Qwen3 文本和 VL 回归测试统一使用 AISBench。
- TTFT 是客户端可见的 time to first token。
- TPOT 是 AISBench 统计的首 token 之后平均每 token 时延。
- `Output Throughput` 是聚合输出吞吐；CSV 里的 `OutputTokenThroughput` 是单请求平均输出吞吐。
- E2EL 是平均请求端到端时延。
- 不同主机负载、模型、上下文或调度器版本下的结果不可直接比较。

## 关键结果

| 方向 | 结果 |
|---|---|
| Paged KV 和 ATB PA | Qwen3-8B F16、10 并发：prefill 吞吐 900-906 tok/s，生成吞吐 85.4-85.9 tok/s，TTFT 3154-3226 ms，TPOT 133-134 ms。profiling 显示 PA 不是主要瓶颈，matmul 数据搬运、norm 和 activation 开销更大。 |
| Prefix cache | Qwen3.5-2B F16，相同 2048-token prompt：全量 prefill TTFT 1994.9 ms；命中 32 个 block 后只处理 64-token suffix 时 TTFT 115.7 ms。 |
| 混合 GDN 融合 prefill | Qwen3.5-2B F16，1024-token prompt、128 输出：recurrent GDN TTFT 2126.5 ms、TPOT 36.1 ms；hybrid fused GDN TTFT 1877-2049 ms、TPOT 36.4 ms。纯 chunk prefill 更快，但有精度漂移，不是默认交付路径。 |
| 调度器 warmup | Qwen3.5-2B F16，1024-token prompt：无 warmup TTFT 1808.6 ms；paged scheduler warmup 后 TTFT 936.7 ms，TPOT 36.1 ms。 |
| ACL graph phase 修复 | Qwen3-1.7B F16、10C 文本：TPOT 从 75.5 ms 降到 73.0 ms，聚合输出吞吐从 127.75 tok/s 提升到 131.97 tok/s。 |
| AISBench 回归 | Qwen3-1.7B F16 10C 文本：TPOT 72.735 ms，输出吞吐 131.395 tok/s，GSM8K 30 样本精度 80.0%。Qwen3-VL-4B F16 图文请求可端到端运行；TextVQA 30 样本精度 96.67%。 |
| VL 回归 | Qwen3-VL-4B F16、单请求：320x320 图片 TTFT 139.7 ms，TPOT 65.0 ms，E2E 8388.9 ms；1280x720 图片 TTFT 416.6 ms，TPOT 83.0 ms，E2E 10954.0 ms。 |

## 时间线总结

| 步骤 | 修改 | 收益或结论 |
|---|---|---|
| 1 | 增加与后端无关的 paged KV cache、block manager、block table、写入 slot 和调度器元数据。 | 支持页粒度 KV 复用和 continuous batching，不把 paging 暴露给模型图。 |
| 2 | 将交错 paged pool 拆成独立 K/V pool。 | 简化布局处理，CANN 可重解释为 ATB FRACTAL_NZ，CPU/CUDA 仍看到 ND pool。 |
| 3 | 接入 CANN/ATB PagedAttention 和 ReshapeCacheOperation。 | attention 和 KV 写入离开 CPU，PA 成为每个 full-attention 层的 paged kernel。 |
| 4 | 优化 PA prefill 调度和 metadata。 | 减少冗余 prefill 和 host metadata 转换，改善 warm TTFT。 |
| 5 | 增加 hybrid paged memory，组合 GDN/linear-attention。 | Qwen3.5 hybrid 使用同一调度器、一致 ubatch 和共享序列计数。 |
| 6 | 增加 block prefix cache 和 GDN checkpoint。 | Qwen3.5-2B 相同前缀 TTFT 从 1994.9 ms 降到 115.7 ms，输出不变。 |
| 7 | 增加 GDN V310 decode 和 chunk prefill 算子。 | GDN 上 NPU；hybrid fused 改善 prefill，纯 chunk 默认关闭。 |
| 8 | 增加 weight NZ layout 和 F16/NZ matmul 快路径。 | 权重只转换一次，减少运行时 transdata。 |
| 9 | 增加 RMSNorm-MUL 融合、flattened RMSNorm 和后端 warmup。 | Qwen3.5-2B TTFT 从 1808.6 ms 降到 936.7 ms。 |
| 10 | 增加 MoE grouped matmul 和 MoE 融合框架。 | 支持 W8A8/grouped MoE 和 NZ 权重复用。 |
| 11 | 把 paged continuous batching 接入 llama-server。 | PA 路径支持多请求、speculative decode、n_cmpl 和 chat API。 |
| 12 | 增加多模态 segment prefill、media prefetch 和 CLIP warmup。 | 首图 TTFT 从 497 ms 降到 317 ms。 |
| 13 | 修复 M-RoPE 多序列 batch positions。 | 修复并发 VL 精度问题，4 并发通过 4/4。 |
| 14 | 为真实 RAC/attention shape 增加 paged scheduler warmup。 | 消除首请求大额冷启动开销。 |
| 15 | 为 paged server decode 增加 ACL graph phase 门控。 | Qwen3-1.7B 10C 文本 TPOT 从 75.5 ms 降到 73.0 ms。 |
| 16 | prefill causal mask 使用 pinned host buffer。 | 避免 graph capture 中 pageable H2D 失败。 |
| 17 | 在 F16/NZ prefill matmul 快路径前探测 `aclnnMatmul`。 | 正常 shape 保留快路径，坏 tiling shape 回退稳定路径。 |
| 18 | 构建可复现 AISBench 性能回归用例。 | text 10C、VL small、VL large 可通过脚本复测。 |
| 19 | 默认开启 mmproj offload。 | VL 默认使用 NPU mmproj；`PERF_LLAMA_MMPROJ_OFFLOAD=0` 回退 CPU。 |

## 详细记录

### 1. Paged KV cache 和 ATB PagedAttention

llama 层记录逻辑 block、序列 metadata 和 paged cache 状态。CANN 后端把 K/V pool 映射为带 FRACTAL_NZ metadata 的 ATB tensor，并通过 ReshapeCacheOperation 写入。模型图只看到 `GGML_OP_PAGED_ATTN`。

Qwen3-8B F16，10 请求，512-token prompt，128 输出：

| 指标 | 结果 |
|---|---:|
| Prefill throughput | 900-906 tok/s |
| Generation throughput | 85.4-85.9 tok/s |
| Average TTFT | 3154-3226 ms |
| Average TPOT | 133-134 ms |

`aclnnMatmul`、RMSNorm、Cast 和 TransData 占据主要窗口；PagedAttention 本身不是首要瓶颈。

### 2. Hybrid paged memory

Hybrid 模型同时包含 full-attention 层和 GDN/linear-attention 层。实现采用 paged KV cache 与 recurrent state 组合，而不是强行塞入同一 memory 抽象。两个子上下文共享 ubatch 和序列顺序。

收益：

- dense 和 hybrid 模型共用 paged scheduler；
- paged metadata 和 recurrent state 不发散；
- 请求完成、回退和 fork 同时释放 KV block 和 recurrent state。

### 3. Prefix cache

prefix cache 使用完整 block 的 chained hash 和引用计数 sharing。hybrid 模型的 GDN state snapshot 保存在 checkpoint slot。

Qwen3.5-2B F16，相同 prompt，2048 个缓存 token：

| 请求类型 | TTFT |
|---|---:|
| 全量 prefill | 1994.9 ms |
| 命中 32 block 加 64-token suffix | 115.7 ms |

cache 开启和关闭时 greedy 输出一致；miss 会退回 suffix prefill。

### 4. Weight NZ 和 matmul

权重复制进 CANN buffer 时一次性转成 FRACTAL_NZ，后续 matmul 直接消费 NZ 权重。

主要收益：

- 减少运行时 transdata；
- dense matmul 和 grouped MoE matmul 共享权重布局；
- 降低 decode 内存流量。

测试负载上，直接关闭 weight NZ 不是受支持回退路径。

### 5. GDN 融合算子

Qwen3.5-2B F16，1024-token prompt，128 输出，greedy：

| 路径 | TTFT | TPOT | 聚合 decode 吞吐 |
|---|---:|---:|---:|
| Recurrent GDN baseline | 2126.5 ms | 36.1 ms | 28.37 tok/s |
| Hybrid fused GDN | 1877-2049 ms | 36.4 ms | 28.10-28.23 tok/s |
| 纯 chunk fused GDN，仅 debug | 1519.8 ms | 37.1 ms | 27.61 tok/s |

hybrid fused GDN 的 prefill 降低约 4-12%，decode 与 baseline 相当，是默认安全路径。纯 chunk 存在精度漂移。

### 6. 调度器 warmup

paged scheduler 在启动阶段执行真实 2-token prefill 加 decode，以及按 `n_ubatch` 构造的 prefill 加 decode，然后释放合成 block。

Qwen3.5-2B F16，1024-token prompt，128 输出：

| 模式 | TTFT | TPOT |
|---|---:|---:|
| 无 warmup | 1808.6 ms | 未测量 |
| paged scheduler warmup | 936.7 ms | 36.1 ms |

### 7. Server continuous batching 和多模态

paged continuous batching 已接入 llama-server，支持 running、swapped、waiting、cache eviction、request priority、speculative decoding、n_cmpl、chat API 和 metrics。

多模态路径：

- prompt 文本和媒体按顺序编码为 segment；
- 调度器执行其他 step 时可 prefetch media embedding；
- server 初始化时预热 CLIP graph。

Qwen3-VL-4B F16 TTFT：

| 场景 | 前 | 后 |
|---|---:|---:|
| 首图 | 497 ms | 317 ms |
| warm image | 392 ms | 312 ms |

修复 M-RoPE batch position 后，4 个并发 VL 请求通过 4/4。

### 8. ACL graph phase 门控

早期 ACL graph 在 Qwen3.5-2B 上能 replay 但性能回退：

| 模式 | TTFT | TPOT |
|---|---:|---:|
| eager | 1181.3 ms | 36.2 ms |
| 早期 graph replay | 4730.3 ms | 58.9 ms |

因此引入 phase gate。server 在 `llama_decode` 前标记 PREFILL、MIXED 或 DECODE，只有纯 decode 进入 ACL graph capture/replay。

Qwen3-1.7B F16，10 请求，约 4104 输入 token，2048 输出：

| 结果 | TTFT | TPOT | E2EL | 聚合输出吞吐 |
|---|---:|---:|---:|---:|
| baseline 1 | 5768.6 ms | 75.5 ms | 160310.8 ms | 127.752 tok/s |
| baseline rerun | 5756.2 ms | 75.3 ms | 159997.6 ms | 128.002 tok/s |
| phase gate 修复 | 5787.0 ms | 73.0 ms | 155150.1 ms | 131.972 tok/s |

### 9. F16 prefill matmul probe

CANN 9.1.0 正式版对大多数 shape 支持 `aclnnMatmul + F16 activation + NZ weight`，但部分 M 维度有 tiling 失败，包括 2048。

当前流程：

1. 先调用 `aclnnMatmulGetWorkspaceSize` 探测；
2. 成功时使用 `aclnnMatmul + F16 + NZ`；
3. 失败时把 activation 转 F32，使用稳定的 `aclnnMm + NZ`；
4. 必要时把输出转回 F16。

### 10. PA/GGML boundary profiling

Qwen3-1.7B F16，单次稳定 decode step，10 请求，block size 128：

| 项目 | 单 step 耗时 | kernel 数 |
|---|---:|---:|
| PagedAttention | 31.62 ms | 28 |
| MatMul | 20.30 ms | 197 |
| ViewCopy | 5.31 ms | 28 |
| TransData | 1.06 ms | 197 |
| Cast | 0.45 ms | 340 |
| 其他可见 kernel | 4.10 ms | mixed |
| host/replay gap | 7.10 ms | 不适用 |

剩余瓶颈集中在 F32/ND 到 F16/NZ 边界、PA 输出转换、小算子融合和 replay 之间的 host 工作。

| 优化方向 | 预期收益 |
|---|---:|
| PA 输出保持 F16 | 约 5.3 ms/step |
| 减少 F32/TND/NZ 边界 | 降低 Cast、TransData、ViewCopy |
| 消除 MatMul boundary transdata | 约 1.0 ms/step |
| 融合 norm/activation/elementwise | 约 2-3 ms/step |
| pipeline sampling 和下一批准备 | 最多约 7 ms/step |

### 11. AISBench 端到端回归

入口是 `tools/server/tests/model-integration/scripts/run_all.py`。text 用例使用精确 4104-token prompt 和 2048 输出；VL 用例使用 320x320 或 1280x720 JPEG，输出 128 token。

Qwen3-1.7B F16，10 请求，CANN 9.1.0 正式版，block size 128：

| 指标 | 结果 |
|---|---:|
| TTFT avg | 6965.791 ms |
| TPOT avg | 72.735 ms |
| E2EL avg | 155854.360 ms |
| Output throughput | 131.395 tok/s |
| GSM8K 30 样本精度 | 80.0% |

Qwen3-VL-4B F16，单请求：

| 场景 | TTFT | TPOT | E2E | 输出吞吐 |
|---|---:|---:|---:|---:|
| 320x320 图片加文本 | 1976.677 ms | 55.616 ms | 9039.868 ms | 14.160 tok/s |
| 1280x720 图片加文本 | 29458.727 ms | 56.683 ms | 36657.463 ms | 3.492 tok/s |

当前 AISBench 版本不把 VL image token 计入吞吐，所以吞吐偏保守。TextVQA 30 样本精度 96.67%，限制生成 16 token，预测以参考答案为前缀即判对。

### 12. 310P CANN 快路径和 Qwen3-VL 兼容修复

#### FP16 prefill matmul

触发条件：

- activation 为 F16；
- query sequence length 大于 1，即 prefill；
- weight 为 FRACTAL_NZ。

优先走 `aclnnMatmul(F16 activation, FRACTAL_NZ weight)`。先调用 `aclnnMatmulGetWorkspaceSize` 做 tiling 探测；成功使用快路径，失败回退：

```text
F16 activation -> F32
aclnnMm(F32 activation, FRACTAL_NZ weight) -> F32 result
F32 result -> F16 output
```

#### FlashAttention mask

| 阶段 | 算子 | mask 接口 |
|---|---|---|
| decode，`q_len == 1` | `aclnnIncreFlashAttentionV4` | additive `pseShift` |
| prefill，`q_len > 1` | `aclnnPromptFlashAttentionV3` | bool `attenMask` |

GGML F16 mask 语义是 `0 = keep`，`-inf = masked`。

decode 允许空 mask：不构造 `pseShift`，IFA 收到 `nullptr`。有显式 mask 时 repeat 到所有 head 后作为 `pseShift`。

prefill 需要 bool `attenMask`。实现分配 `K x K` bool mask，先填充 `false`，再把显式 F16 mask 的 `q_len x K` 前缀 cast 成 bool。

安全门控：310P PFA prefill 加空 mask 在 full attention 上出现过数值错误，因此 `supports_op` 拒绝 `q_len > 1 && mask == nullptr`，回退 CPU softmax；空 mask decode 仍在 NPU。

#### Qwen3-VL RoPE

Qwen3-VL 使用 M-RoPE/IMROPE 和 vision RoPE。旧断言拒绝 M-RoPE/IMROPE，现在接受 `full-width || vision || M-RoPE/IMROPE`。

`Qcur/Kcur` 是 QKV 拆分后的 view，在 `ggml_rope_multi` 前复制为 contiguous buffer，满足 310P partial RoPE 输入约束。

#### 其他修复

1. **bilinear upscale**：接入 `aclnnUpsampleBilinear2d`，与 nearest 按模式分派，消除 Qwen3-VL warmup 的最后一个 unsupported op。
2. **dense weight NZ**：dense 权重不再强制 F16 才能成为候选，但仍排除 BF16、W8A8、非 contiguous 和未 32 对齐 tensor；expert weight 仍限制 F16。
3. **pinned causal mask**：host mask 改用 `aclrtMallocHost`，避免 graph capture 中的 pageable H2D；尺寸变化时先同步 stream 再释放旧 buffer。

## vLLM Ascend 对比

### Qwen3-1.7B 文本，10 请求

| Engine | TTFT avg | TPOT avg | E2EL avg | 聚合输出吞吐 |
|---|---:|---:|---:|---:|
| llama.cpp | 5816.5 ms | 69.5 ms | 148180.8 ms | 138.179 tok/s |
| vLLM Ascend | 5733.6 ms | 63.8 ms | 136398.0 ms | 149.691 tok/s |

历史结果输入长度略有差异：llama.cpp 平均 4104 token，vLLM 平均 4317.5 token。剩余差距主要在稳定 decode 时延和 host 侧工作。

### Qwen3-VL-4B，单请求

| 场景 | Engine | TTFT | TPOT | E2EL | 聚合输出吞吐 |
|---|---|---:|---:|---:|---:|
| 320x320 图片 | llama.cpp | 139.7 ms | 65.0 ms | 8388.9 ms | 15.258 tok/s |
| 320x320 图片 | vLLM Ascend | 145.8 ms | 59.2 ms | 7667.2 ms | 16.695 tok/s |
| 1280x720 图片 | llama.cpp | 416.6 ms | 83.0 ms | 10954.0 ms | 11.685 tok/s |
| 1280x720 图片 | vLLM Ascend | 356.8 ms | 78.4 ms | 10309.2 ms | 12.416 tok/s |

VL small TTFT 已接近 vLLM Ascend，剩余差距主要在 decode，尤其大图路径。

## 负结果

| 实验 | 结果 |
|---|---|
| 直接 F16 RMSNorm | 输出错误，prefill 吞吐下降。 |
| 三输入 AddRmsNorm + Mul 融合 | 真实负载语义漂移。 |
| ubatch 2048 | 超过测试模型 compute buffer。 |
| `cubeMathType=3` | 310P F32 cube 不支持。 |
| `cubeMathType=2` | prefill 无稳定收益。 |
| 512 MB F16 input cache | 相比 256 MB 无改善。 |
| upstream ADD + RMS_NORM 融合 | 在波动范围内。 |
| `aclnnFusedMatmul` | cast 后没有稳定端到端收益。 |
| 临时 F16 matmul 输出加 F32 cast | 10 请求负载无稳定收益。 |
| 纯 chunk GDN state | prefill 更快，但精度漂移。 |
| 无 phase gating 的早期 ACL graph replay | TTFT 和 TPOT 回退。 |
| `aclnnMatmulWeightNz` | 测试过的 310P 构建不支持。 |

## 后续工作

1. 保持 PA 输出为 F16，消除 ViewCopy/Cast 往返。
2. 在 graph 允许范围内稳定 decode activation 为 F16。
3. 复用 F16/NZ activation，减少 MatMul boundary transdata。
4. 融合 RoPE、QK norm、residual norm 和 SwiGLU。
5. 把 sampling、logits 处理和 next-batch preparation 移出关键 decode 路径。
6. 增加 chunked-prefill scheduling，避免长 prompt 阻塞 decode batch。
7. 每次性能修改继续使用 AISBench 回归验证。

## 相关文件

- `tools/server/tests/model-integration/README.md`
- `tools/server/tests/model-integration/scripts/run_all.py`
- `docs/backend/CANN-Hybrid-PagedAttention-Design.md`
- `docs/backend/CANN-Hybrid-PagedAttention-PrefixCache-Design.md`
- `docs/design/Paged-Server-Integration-Design.md`
