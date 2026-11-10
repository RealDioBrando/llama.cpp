# develop_paged_integration Change Summary

## Scope

Branch: `develop_paged_integration`

Baseline: `gitcode/develop`

Code diff:

```text
34 files changed, 1820 insertions(+), 26 deletions(-)
```

## 1. CANN Backend Changes

### 1.1 F16 prefill matmul fast path

File: `ggml/src/ggml-cann/aclnn_ops.cpp`

- Added a 310P fast path for FP16 prefill when `input->ne[1] > 1`.
- Preferred path: `aclnnMatmul + F16 activation + FRACTAL_NZ weight`.
- Because CANN tiling rejects some `M + mat2 shape` combinations, the executor
  is probed with `aclnnMatmulGetWorkspaceSize` before execution.
- Stable fallback: `F16 -> F32 + aclnnMm + NZ -> F16`.
- `M=2048` is not an independent failure condition. It is tied to the right
  matrix shape/layout. For example, `M=2048, K=2560, N=4096` fails, while some
  other NZ shapes pass.

### 1.2 310P FlashAttention mask semantics

File: `ggml/src/ggml-cann/aclnn_ops.cpp`

- Allow an empty mask for 310P FlashAttention.
- Prefill uses an all-keep mask.
- Decode does not pass pseShift.
- Avoids an assertion failure for models and graph paths without an explicit mask.

### 1.3 Bilinear upsample

Files:

- `ggml/src/ggml-cann/aclnn_ops.cpp`
- `ggml/src/ggml-cann/aclnn_ops.h`
- `ggml/src/ggml-cann/ggml-cann.cpp`

- Added `aclnnUpsampleBilinear2d`.
- CANN `GGML_OP_UPSCALE` now routes nearest and bilinear scale modes correctly.

### 1.4 Partial RoPE and vision RoPE fixes

Files:

- `ggml/src/ggml-cann/aclnn_ops.cpp`
- `tools/mtmd/models/qwen3vl.cpp`

- Enter the 310P composite RoPE path only when `n_dims == src0->ne[0]`.
- Permit vision RoPE in the relevant assertion.
- Make Qwen3-VL `Qcur/Kcur` contiguous before M-RoPE to satisfy 310P partial
  RoPE input requirements.

### 1.5 Pinned prefill causal mask

File: `ggml/src/ggml-cann/paged_attn.cpp`

- Replaced pageable host memory with pinned host memory for the prefill causal
  mask.
- Prevents pageable H2D copies during ACL graph capture.
- Added allocation, reuse, and release handling for the pinned buffer.

### 1.6 Dense weight NZ candidates

File: `ggml/src/ggml-cann/weight_layout.cpp`

- Dense-weight NZ selection no longer requires F16 only.
- BF16, W8A8, quantized special paths, non-contiguous tensors, and tensors
  without 32-aligned dimensions remain excluded.

## 2. Paged Scheduler and Server Integration

### 2.1 Waiting queue row budget

File: `src/llama-paged-scheduler-impl.cpp`

- Waiting prefill processing now decrements `row_budget` correctly.
- Prevents waiting requests from exceeding the batch token budget.

### 2.2 Prefill accounting

File: `src/llama-paged-scheduler-impl.cpp`

- `n_prompt` is decremented by the actual consumed batch length in scheduler
  `update()`.
- Fixes prefill completion detection and the logits sampling point.

### 2.3 ACL graph phase gate

File: `tools/server/server-scheduler-paged.cpp`

- Compute graph phase before `llama_decode`.
- Distinguish `PREFILL`, `MIXED`, and `DECODE`.
- Only stable decode enters ACL graph capture/replay.
- Keep prefill and mixed batches eager so changing shapes and prefill masks do
  not enter graph capture.

### 2.4 Multimodal warmup switch

File: `tools/server/server-scheduler-paged.cpp`

- Added `LLAMA_PAGED_MM_WARMUP=0` to disable synthetic-image CLIP warmup.
- Useful for restricted environments and accuracy runs where unsupported vision
  RoPE shapes must be bypassed.

## 3. Model Integration Tests

Directory: `tools/server/tests/model-integration/`

Features:

- Unified entry point for llama.cpp and vLLM Ascend.
- Covers both performance and accuracy.
- Model paths, tokenizer paths, ports, NPU IDs, and build directories are
  supplied through environment variables.
- Supports AISBench performance cases and GSM8K / TextVQA accuracy cases.
- Supports 30-sample and full accuracy datasets.
- Generates performance summaries, accuracy summaries, and comparison reports.

### 3.1 Test scenarios

| Scenario | Model | Workload |
|---|---|---|
| `text_10c` | Qwen3-1.7B F16 | 10 concurrency, about 4096 input tokens, 2048 output tokens |
| `vl_small` | Qwen3-VL-4B F16 | 320x320 image plus text, 128 output tokens |
| `vl_large` | Qwen3-VL-4B F16 | 1280x720 image plus text, 128 output tokens |
| Text accuracy | Qwen3-1.7B F16 | GSM8K, default 30 samples |
| VL accuracy | Qwen3-VL-4B F16 | TextVQA, default 30 samples |

### 3.2 Common commands

Build llama.cpp:

```bash
scripts/build_llama.sh
```

Prepare base image assets:

```bash
python scripts/prepare_assets.py
```

Prepare accuracy assets:

```bash
python scripts/prepare_accuracy_assets.py
```

Run all llama.cpp performance cases:

```bash
python scripts/run_all.py --engine llama --scenario all
```

Run all vLLM baseline cases:

```bash
python scripts/run_all.py --engine vllm --scenario all --save-baseline
```

Run accuracy only:

```bash
python scripts/run_all.py --engine llama --scenario text_10c --accuracy-mode only
python scripts/run_all.py --engine llama --scenario vl_small --accuracy-mode only
```

Run full accuracy datasets:

```bash
python scripts/run_all.py --engine llama --scenario all --accuracy-mode only --accuracy-full
```

## 4. Test Verification

Test environment:

```text
Ascend 310P3
CANN 9.1.0 official
Single NPU
Qwen3 / Qwen3-VL F16 GGUF
```

### 4.1 Build and basic tests

- CANN aarch64 build passes.
- `llama-server` builds.
- Paged-related targets build.
- `test-paged-kv-e2e` passes after scheduler warmup.
- Default server regression passes.

Covered interfaces:

```text
/completion
/v1/completions
/v1/chat/completions
/metrics
/props
/tokenize
/slots
```

Covered behavior:

- Non-streaming requests.
- Streaming SSE.
- Stop strings.
- EOG termination.
- Sampling parameters.
- Deterministic output with `temperature=0`.
- Request cancellation.
- Concurrent queueing.
- Tool calls.
- JSON schema output.
- `n > 1` parallel sampling.

### 4.2 Paged server tests

Verified:

- Paged KV initialization.
- Continuous batching scheduling.
- Prefill / decode phase separation.
- ACL graph decode capture / replay.
- Prefix cache hits.
- Cross-request block reuse.
- Request cancellation and KV release.
- Swap and recompute paths under GPU block pressure.
- Speculative partial acceptance.
- With `n_draft = 2/3/4/6/8`, 128-token greedy outputs match the non-speculative
  baseline.

### 4.3 Multimodal tests

Verified:

- Qwen3-VL image-and-text requests run on the paged path.
- `test_paged_server_mm.py` passes all 3 cases.
- Image context is understood correctly.
- Prompt token accounting is correct.
- Multi-sequence M-RoPE concurrency cases pass after the position fix.
- CLIP warmup and media prefetch paths work.
- `LLAMA_PAGED_MM_WARMUP=0` escape switch works.

### 4.4 AISBench performance results

| Scenario | Model | TTFT avg | TPOT avg | E2E / E2EL avg | Output throughput |
|---|---|---:|---:|---:|---:|
| `text_10c` | Qwen3-1.7B F16 | 6965.791 ms | 72.735 ms | 155854.360 ms | 131.395 tok/s |
| `vl_small` | Qwen3-VL-4B F16 | 1976.677 ms | 55.616 ms | 9039.868 ms | 14.160 tok/s |
| `vl_large` | Qwen3-VL-4B F16 | 29458.727 ms | 56.683 ms | 36657.463 ms | 3.492 tok/s |

### 4.5 Accuracy results

| Dataset | Samples | Accuracy |
|---|---:|---:|
| GSM8K | 30 | 80.0% |
| TextVQA | 30 | 96.67% |

TextVQA details:

```text
29 / 30 samples pass
30 / 30 requests succeed
```

## 5. Conclusion

- Functional, accuracy, and regression cases pass.
- The performance cases provide a stable re-test baseline.
- Current measured performance has not yet reached all business targets in the
  README, so the results should not be described as all performance targets
  being met.
