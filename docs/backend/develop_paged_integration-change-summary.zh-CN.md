# `develop_paged_integration` 分支修改说明

## 1. 范围与基线

- 分支：`develop_paged_integration`
- 基线：`origin/develop`
- 当前参考提交：`9573c3b3c`
- 修改规模：20 个提交，34 个文件变更，`+1844 / -43`

本分支的目标是把 PagedAttention、连续批处理、CANN 310P 算子适配和多模态能力接入 `llama-server`，并补齐性能与精度回归测试入口。

## 2. 主要修改

### 2.1 CANN 算子适配

修改集中在 `ggml/src/ggml-cann/`：

1. **FP16 prefill matmul 快路径**
   - 310P 上优先尝试 `aclnnMatmul + FP16 activation + FRACTAL_NZ weight`。
   - 执行前通过 `aclnnMatmulGetWorkspaceSize` 探测 tiling 是否可用。
   - 探测失败时回退到 `FP16 -> F32 + aclnnMm + NZ -> F16` 稳定路径。
   - 移除了只在 CANN 9.1.0-beta.1 上需要的旧 workaround。

2. **310P FlashAttention mask 语义修复**
   - 允许 prefill 阶段使用空 mask。
   - 无显式 mask 的 full attention prefill 在 310P 上回退到 CPU softmax，避免 PFA 数值错误。
   - prefill 使用 all-keep mask，decode 阶段不再传入 `pseShift`。

3. **RoPE 适配**
   - 放行 Qwen3-VL 使用的 mRoPE / IMROPE 路径。
   - 放行 vision RoPE 的全宽旋转路径。
   - Qwen3-VL 的 `Qcur/Kcur` 在进入 M-RoPE 前转成 contiguous，满足 310P partial RoPE 输入约束。

4. **其他算子补齐**
   - 接入 `aclnnUpsampleBilinear2d`，支持 bilinear upscale。
   - 扩展 dense weight 的 NZ 候选范围，保留 BF16、W8A8、非连续张量等排除条件。
   - prefill causal mask 改为 pinned host memory，避免 ACL graph capture 阶段出现 pageable H2D。

### 2.2 调度器与 Server 集成

修改集中在 `src/llama-paged-scheduler-impl.cpp` 和 `tools/server/server-scheduler-paged.cpp`：

1. **调度正确性**
   - 修正 waiting prefill 处理时没有正确扣减 `row_budget` 的问题。
   - 修正 `n_prompt` 的扣减逻辑，保证 prefill 完成判定和 logits 采样点正确。

2. **执行阶段控制**
   - 在 `llama_decode` 前显式区分 `PREFILL`、`MIXED`、`DECODE`。
   - 只有稳定 decode 阶段进入 ACL graph capture/replay。
   - prefill 和 mixed batch 保持 eager，避免动态 shape 和 prefill mask 进入 graph capture。

3. **多模态路径**
   - 图文请求走 paged scheduler 路径。
   - 增加媒体编码预取和 CLIP warmup，降低图文 TTFT。
   - 提供 `LLAMA_PAGED_MM_WARMUP=0` 开关，可在受限环境或精度测试中关闭合成图 warmup。

### 2.3 模型适配

- `tools/mtmd/models/qwen3vl.cpp`
  - 将 Qwen3-VL ViT 的 Q/K 在 partial RoPE 前转成 contiguous。
  - 修复图像编码在 CANN 310P 上的算子路由和精度问题。

### 2.4 测试与工具链

新增 `tools/server/tests/model-integration/`：

- 统一接入 llama.cpp 和 vLLM Ascend。
- 支持 AISBench 性能测试。
- 支持 GSM8K、TextVQA 精度测试。
- 支持 30 样本和全量数据集。
- 自动生成性能汇总、精度汇总和对比报告。
- 模型路径、tokenizer、端口、NPU ID、构建目录均通过环境变量注入。

## 3. 测试情况

### 3.1 测试环境

```text
Ascend 310P3
CANN 9.1.0 正式版
单卡
Qwen3 / Qwen3-VL F16 GGUF
```

### 3.2 功能与精度验证

已验证内容包括：

- paged KV 初始化。
- 连续批处理调度。
- prefill / decode 阶段分离。
- ACL graph decode capture / replay。
- prefix cache 命中。
- 跨请求 block 复用。
- 请求取消与 KV 释放。
- GPU block 压力下的 swap / recompute。
- speculative partial acceptance。
- `temperature=0` 下的确定性输出。
- streaming / non-streaming 请求。
- stop string、EOG、并发排队、工具调用、JSON schema 输出、`n > 1` 并行采样。

多模态验证：

- Qwen3-VL 图文请求可以在 paged 路径执行。
- `test_paged_server_mm.py` 相关用例通过。
- 图像上下文理解正常。
- prompt token 统计正确。
- 多序列 M-RoPE 并发用例在位置修复后通过。
- CLIP warmup 和媒体预取路径工作正常。
- `LLAMA_PAGED_MM_WARMUP=0` 开关有效。

### 3.3 精度结果

| 数据集 | 样本数 | 精度 |
|---|---:|---:|
| GSM8K | 30 | 80.0% |
| TextVQA | 30 | 96.67% |

TextVQA 细节：

```text
29 / 30 样本通过
30 / 30 请求成功
```

### 3.4 性能结果

性能回归套件已有以下历史基线：

| 场景 | 模型 | TTFT avg | TPOT avg | E2E / E2EL avg | 输出吞吐 |
|---|---|---:|---:|---:|---:|
| `text_10c` | Qwen3-1.7B F16 | 6965.791 ms | 72.735 ms | 155854.360 ms | 131.395 tok/s |
| `vl_small` | Qwen3-VL-4B F16 | 1976.677 ms | 55.616 ms | 9039.868 ms | 14.160 tok/s |
| `vl_large` | Qwen3-VL-4B F16 | 29458.727 ms | 56.683 ms | 36657.463 ms | 3.492 tok/s |

针对最近的多模态修复，另做了 targeted 复测：

| 场景 | TTFT | TPOT | 输出吞吐 |
|---|---:|---:|---:|
| `vl_small` | 339.1 ms | 56.0 ms | 17.17 tok/s |
| `vl_large` | 2520.2 ms | 57.5 ms | 13.02 tok/s |

说明：上面两个表分别对应回归套件的历史基线和最近一次针对性复测，二者运行环境和负载定义相同，但后者的输入场景为多模态 targeted 用例。

### 3.5 常用测试命令

构建：

```bash
scripts/build_llama.sh
```

准备基础图像资源：

```bash
python scripts/prepare_assets.py
```

准备精度资源：

```bash
python scripts/prepare_accuracy_assets.py
```

跑 llama.cpp 全量性能用例：

```bash
python scripts/run_all.py --engine llama --scenario all
```

只跑精度：

```bash
python scripts/run_all.py --engine llama --scenario vl_small --accuracy-mode only
python scripts/run_all.py --engine llama --scenario text_10c --accuracy-mode only
```

跑全量精度数据集：

```bash
python scripts/run_all.py --engine llama --scenario all --accuracy-mode only --accuracy-full
```

## 4. 结论

- 功能、精度和回归用例已经打通。
- 多模态 TTFT 的关键 CPU 回退点已经收敛到可控范围。
- `GGML_CANN_WEIGHT_NZ=1` 下 Qwen3-VL 精度保持正常，因此不依赖强制关闭 NZ。
- 当前性能结果可以作为后续对比基线，但尚未全部达到业务目标，不应描述为性能目标全部达成。
