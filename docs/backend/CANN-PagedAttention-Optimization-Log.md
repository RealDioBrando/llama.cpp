# CANN PagedAttention optimization log

Status: engineering log

Last updated: 2026-11-10

This file records the CANN PagedAttention, hybrid GDN, server integration, and
ACL graph work in one place. It is a working log, not a design proposal. Each
entry records the change, the observed benefit, and the evidence. The newest
entries are at the end of the timeline.

## Scope and conventions

- Device: Ascend 310P3 unless noted.
- Software: CANN 9.1.0 official build, ATB PagedAttention.
- AISBench metrics are used for Qwen3 text and VL regression tests.
- TTFT is the client-visible time to first token.
- TPOT is the AISBench average time per output token after the first token.
- `Output Throughput` is the aggregate generated-token throughput reported by
  AISBench. `OutputTokenThroughput` in the CSV is per-request average throughput.
- E2EL is the average request end-to-end latency.
- Results from different host load, model, context, or scheduler versions are
  not directly comparable unless the workload and build are also the same.

## Snapshot

The most important verified results are:

| Area | Result |
|---|---|
| Paged KV and ATB PA | Qwen3-8B F16, 10 requests: prefill 900-906 tok/s, generation 85.4-85.9 tok/s, TTFT 3154-3226 ms, TPOT 133-134 ms. Profiling shows PA is not the bottleneck; matmul data movement and norm/activation are larger costs. |
| Prefix cache | Qwen3.5-2B F16, same 2048-token prompt: full-prefill TTFT 1994.9 ms; 32-block hit plus 64-token suffix prefill TTFT 115.7 ms. Greedy outputs match with cache off/on. |
| Hybrid GDN fused prefill | Qwen3.5-2B F16, 1024-token prompt and 128 output: recurrent GDN TTFT 2126.5 ms, TPOT 36.1 ms; hybrid fused GDN TTFT 1877-2049 ms, TPOT 36.4 ms. Pure chunk is faster in prefill but has accuracy drift and is not a delivery path. |
| Scheduler warmup | Qwen3.5-2B F16, 1024-token prompt: no warmup TTFT 1808.6 ms; paged scheduler warmup TTFT 936.7 ms, TPOT 36.1 ms. |
| ACL graph phase fix | Qwen3-1.7B F16, 10C text: TPOT 75.5 to 73.0 ms, aggregate output throughput 127.75 to 131.97 tok/s, E2E average 160310.8 to 155150.1 ms. Server logs confirm capture followed by stable replay. |
| AISBench regression | Qwen3-1.7B F16 10C text: TPOT 72.735 ms and 131.395 tok/s output with 80.0% GSM8K accuracy on 30 samples. Qwen3-VL-4B F16 image requests run end-to-end; TextVQA 30 samples reaches 96.67% with the VQA prefix evaluator. |
| VL regression | Qwen3-VL-4B F16, single request: 320x320 image has TTFT 139.7 ms, TPOT 65.0 ms, E2E 8388.9 ms; 1280x720 image has TTFT 416.6 ms, TPOT 83.0 ms, E2E 10954.0 ms. |

## Timeline summary

| Step | Change | Benefit or conclusion |
|---|---|---|
| 1 | Add backend-independent paged KV cache, block manager, block table, write slots, and scheduler metadata. | Enables page-granular KV reuse and continuous batching without exposing paging to the model graph. |
| 2 | Split the interleaved paged pool into separate K/V pools. | Simplifies layout handling; CANN can reinterpret the pools as ATB FRACTAL_NZ while CPU/CUDA still see ND pools. |
| 3 | Add CANN/ATB PagedAttention and ReshapeCacheOperation. | Moves attention and KV writes off CPU; PA becomes one paged kernel per full-attention layer. |
| 4 | Optimize PA prefill dispatch and metadata. | Reduces redundant prefill work and host metadata conversion; improves warm-state TTFT. |
| 5 | Add hybrid paged memory and GDN/linear-attention composition. | Qwen3.5 hybrid models run with one scheduler, consistent ubatch, and shared sequence accounting. |
| 6 | Add block-prefix cache and GDN checkpoint. | Same-prefix Qwen3.5-2B request TTFT drops from 1994.9 ms to 115.7 ms; greedy output stays unchanged. |
| 7 | Add GDN V310 decode and chunk prefill operators. | Enables GDN on NPU. Hybrid fused mode improves TTFT by about 4-12%; pure chunk is not default because state precision drifts. |
| 8 | Add weight NZ layout and F16/NZ matmul fast path. | Converts weights once at load time, removes repeated runtime transdata, and improves matmul efficiency. |
| 9 | Add RMSNorm-MUL fusion, flattened RMSNorm, and backend warmup. | Removes small host/device overhead; warmup reduces TTFT from 1808.6 ms to 936.7 ms on Qwen3.5-2B. |
| 10 | Add MoE grouped matmul and MoE fusion framework. | Enables W8A8/grouped MoE on CANN and reuses NZ weights; not all fusion knobs show stable E2E gain. |
| 11 | Integrate paged continuous batching into llama-server. | Provides server-level multi-request, speculative decode, n_cmpl, and chat API support on the PA path. |
| 12 | Add multimodal segment prefill, media prefetch, and CLIP warmup. | First-image TTFT improves from 497 ms to 317 ms; warm-image TTFT improves from 392 ms to 312 ms. |
| 13 | Fix M-RoPE multi-sequence batch positions. | Fixes concurrent VL correctness; 4-concurrency passes 4/4 and aggregate throughput reaches 56.5 tok/s. |
| 14 | Add paged scheduler warmup for the real RAC/attention shape. | Removes the large first-request penalty without leaving synthetic blocks in the scheduler. |
| 15 | Gate ACL graph phase for paged server decode. | Qwen3-1.7B 10C text TPOT improves 75.5 to 73.0 ms and aggregate output throughput improves 127.75 to 131.97 tok/s. |
| 16 | Add pinned host causal-mask buffer. | Avoids pageable H2D in graph capture and removes the prefill mask capture failure. |
| 17 | Probe `aclnnMatmul` before the F16/NZ prefill fast path. | Keeps the fast path on CANN 9.1.0 official builds and falls back to stable `aclnnMm + NZ` for the M=2048 tiling case. |
| 18 | Build reproducible AISBench perf regression cases. | Text 10C, VL small, and VL large are now command-driven and use environment-supplied model paths. |
| 19 | Enable mmproj offload by default. | VL test cases use NPU mmproj offload unless `PERF_LLAMA_MMPROJ_OFFLOAD=0` is set for CPU fallback debugging. |

## Detailed records

### 1. Paged KV cache and ATB PagedAttention

The paged cache records logical blocks and per-sequence metadata in the llama
layer. The CANN backend maps the K/V pool pointers to ATB tensors with
FRACTAL_NZ metadata and writes them through ReshapeCacheOperation. The graph
only sees `GGML_OP_PAGED_ATTN`, not the NZ layout.

Qwen3-8B F16 on Atlas 300I Duo 310P, 10 requests, 512-token prompt, 128 output:

| Metric | Result |
|---|---:|
| Prefill throughput | 900-906 tok/s |
| Generation throughput | 85.4-85.9 tok/s |
| Average TTFT | 3154-3226 ms |
| Average TPOT | 133-134 ms |

Profiling shows that `aclnnMatmul`, RMSNorm, Cast, and TransData dominate the
window. PagedAttention itself is not the primary bottleneck.

### 2. Hybrid paged memory

Hybrid models have both full-attention layers and GDN/linear-attention layers.
The implementation composes a paged KV cache with recurrent state instead of
forcing both into one memory abstraction. Both subcontexts see the same ubatch
and sequence order.

Benefits:

- dense and hybrid models can share the same paged scheduler;
- paged metadata and recurrent state no longer diverge;
- request finish, rollback, and fork release both KV blocks and recurrent state.

### 3. Prefix cache

The prefix cache uses a chained hash over complete blocks and ref-counted block
sharing. For hybrid models, GDN state snapshots are stored at checkpoint slots.

Qwen3.5-2B F16, same prompt, 2048 cached tokens:

| Request type | TTFT |
|---|---:|
| Full prefill | 1994.9 ms |
| 32-block prefix hit plus 64-token suffix | 115.7 ms |

Greedy output is identical with the cache off and on. The cache is safe to miss
because the fallback is ordinary suffix prefill.

### 4. Weight NZ and matmul

Weights are converted to FRACTAL_NZ once when copied into the CANN buffer. The
matmul path can then consume NZ weights directly instead of converting weights
per step.

The main benefits are:

- fewer runtime transdata kernels;
- one weight layout shared by dense and grouped MoE matmul;
- lower device memory traffic in decode.

Disabling the weight NZ path is not a supported fallback because the runtime
allocation/conversion path can fail on the tested workloads.

### 5. GDN fused operators

Qwen3.5-2B F16, 1024-token prompt, 128 output, greedy:

| Path | TTFT | TPOT | Aggregate decode throughput |
|---|---:|---:|---:|
| Recurrent GDN baseline | 2126.5 ms | 36.1 ms | 28.37 tok/s |
| Hybrid fused GDN | 1877-2049 ms | 36.4 ms | 28.10-28.23 tok/s |
| Pure chunk fused GDN, debug only | 1519.8 ms | 37.1 ms | 27.61 tok/s |

Hybrid fused GDN reduces prefill by about 4-12% and keeps decode within the
baseline range. It is the default-safe path. Pure chunk is faster in prefill
but has state precision drift and is not enabled for delivery.

### 6. Scheduler warmup

The paged scheduler performs a real 2-token prefill plus decode and an
`n_ubatch`-shaped prefill plus decode at startup, then releases the synthetic
blocks.

Qwen3.5-2B F16, 1024-token prompt, 128 output:

| Mode | TTFT | TPOT |
|---|---:|---:|
| No paged warmup | 1808.6 ms | not measured |
| Paged scheduler warmup | 936.7 ms | 36.1 ms |

Warmup is especially important for alternating prefill/decode shapes because it
removes first-call CANN, ATB tiling, and ReshapeCacheOperation setup.

### 7. Server continuous batching and multimodal

Paged continuous batching was integrated into llama-server. This added running,
swapped, waiting, cache eviction, request priority, speculative decoding,
n_cmpl, chat APIs, and metrics to the PA path.

For multimodal requests:

- prompt text and media are encoded as ordered segments;
- media embeddings can be prefetched while the scheduler runs another step;
- CLIP graphs are warmed during server init.

Qwen3-VL-4B F16 TTFT:

| Case | Before prefetch/warmup | After prefetch/warmup |
|---|---:|---:|
| First image | 497 ms | 317 ms |
| Warm image | 392 ms | 312 ms |

M-RoPE batch position fixes made concurrent VL runs correct. After the fix, 4
concurrent requests passed 4/4 and aggregate throughput reached 56.5 tok/s.

### 8. ACL graph phase gate

Earlier ACL graph experiments could replay on Qwen3.5-2B but were a regression:

| Mode | TTFT | TPOT |
|---|---:|---:|
| Eager | 1181.3 ms | 36.2 ms |
| Early ACL graph replay | 4730.3 ms | 58.9 ms |

That negative result led to the phase gate work. The current paged server path
now marks the batch as PREFILL, MIXED, or DECODE before `llama_decode`. Only
pure decode enters ACL graph capture/replay. The prefill causal mask also uses
pinned host memory, so capture does not fail on pageable H2D copies.

Qwen3-1.7B F16, 10 requests, about 4104 input tokens, 2048 output tokens:

| Run | TTFT avg | TPOT avg | E2EL avg | Aggregate output throughput |
|---|---:|---:|---:|---:|
| 2026-11-09 baseline 1 | 5768.6 ms | 75.5 ms | 160310.8 ms | 127.752 tok/s |
| 2026-11-09 baseline rerun | 5756.2 ms | 75.3 ms | 159997.6 ms | 128.002 tok/s |
| Phase-gate and pinned mask fix | 5787.0 ms | 73.0 ms | 155150.1 ms | 131.972 tok/s |

Compared with the two same-day baselines, the fix improves TPOT by about
2.3-2.5 ms, E2EL by about 4.8-5.2 seconds, and aggregate output throughput by
about 4.0 tok/s. TTFT is unchanged within run variance.

Debug logs showed the first decode graph capture followed by stable replay. This
confirmed that the improvement was actual ACL graph replay, not the earlier
llama-level graph topology reuse counter.

### 9. F16 prefill matmul probe

CANN 9.1.0 official builds support `aclnnMatmul` with F16 activation and NZ
weight for most shapes, but a known tiling failure occurs at M values including
2048. The current implementation probes `aclnnMatmulGetWorkspaceSize` first:

1. if the probe succeeds, use `aclnnMatmul + F16 + NZ`;
2. if it fails, cast activation to F32 and use stable `aclnnMm + NZ`;
3. cast the output back to F16 if needed.

This keeps the F16/NZ fast path while avoiding the M=2048 warmup and prefill
tiling abort.

### 10. PA/GGML boundary profiling

One Qwen3-1.7B F16 stable decode step, 10 requests, CANN 9.1.0, block size 128:

| Item | Time per step | Kernel count |
|---|---:|---:|
| PagedAttention | 31.62 ms | 28 |
| MatMul | 20.30 ms | 197 |
| ViewCopy | 5.31 ms | 28 |
| TransData | 1.06 ms | 197 |
| Cast | 0.45 ms | 340 |
| Other visible kernels | 4.10 ms | mixed |
| Host/replay gap | 7.10 ms | not applicable |

This says the remaining opportunities are not PA itself. They are the F32/ND to
F16/NZ bridge, PA output conversion, small op fusion, and host work between
graph replays. The candidate gains are:

| Candidate | Expected gain |
|---|---:|
| Keep PA output in F16 | about 5.3 ms/step from ViewCopy and Cast |
| Reduce F32/TND/NZ boundary crossings | lower Cast, TransData, and ViewCopy |
| Remove MatMul boundary transdata | about 1.0 ms/step |
| Fuse small norm/activation/elementwise ops | about 2-3 ms/step |
| Pipeline sampling and next-batch preparation | up to about 7 ms/step |

### 11. AISBench end-to-end regression

The model integration entry is `tools/server/tests/model-integration/scripts/run_all.py`. It starts
the llama-server, runs AISBench, and writes aggregate results under
`tools/server/tests/model-integration/results`. The text case uses an exact 4,104-token prompt
and 2,048 output tokens. The VL cases use a generated 320x320 or 1280x720
JPEG and 128 output tokens. The VL server enables mmproj offload by default.
Historical baseline runs used `--no-mmproj-offload` because the earlier CANN
build aborted in GPU CLIP RoPE; the vision RoPE fix is recorded in section 12.

Qwen3-1.7B F16, 10 requests, CANN 9.1.0 official build, block size 128:

| Metric | AISBench result |
|---|---:|
| TTFT avg | 6965.791 ms |
| TPOT avg | 72.735 ms |
| E2EL avg | 155854.360 ms |
| Output throughput | 131.395 tok/s |
| GSM8K accuracy, 30 samples | 80.0% |

The phase-gate and pinned-mask fixes are the main graph-related changes in
this comparison. Against the two same-day eager baselines, TPOT improves by
2.3-2.8 ms and aggregate output throughput improves by about 3.6 tok/s.

Qwen3-VL-4B F16, single request:

| Case | TTFT avg | TPOT avg | E2E avg | Output tok/s |
|---|---:|---:|---:|---:|
| 320x320 image + text | 1976.677 ms | 55.616 ms | 9039.868 ms | 14.160 |
| 1280x720 image + text | 29458.727 ms | 56.683 ms | 36657.463 ms | 3.492 |

The VL Image tokens are not counted by AISBench in this version, so the
throughput columns are conservative. TextVQA 30-sample accuracy is 96.67%
(29 of 30 samples pass; 30/30 requests succeed). The TextVQA case limits
generation to 16 tokens and treats a prediction as correct when the normalized
prediction starts with one of the reference answers. This removes the false
failures from verbose tails and repeated text while still requiring the
reference answer to be the response prefix.

### 12. 310P CANN fast-path and Qwen3-VL compatibility fixes

These changes keep supported operators on NPU while adding explicit fallbacks
where CANN 310P behavior is unsafe.

#### FP16 prefill matmul fast path

The fast path applies to Ascend 310P when all of these are true:

- the activation tensor is F16;
- the query sequence length is greater than 1, which means prefill;
- the weight is FRACTAL_NZ.

The preferred path is `aclnnMatmul` with F16 activation and FRACTAL_NZ weight.
It avoids the two extra casts and the temporary F32 tensor used by the older
stable route.

CANN 9.1.0 official builds reject some M dimensions during matmul tiling. A
known example is `M=2048`, which is created by a `-b 2048 -ub 2048` warmup. To
avoid aborting the graph, the implementation calls
`aclnnMatmulGetWorkspaceSize` before executing the op. This is a tiling probe,
not an NPU compute call. If the probe succeeds, the implementation uses
`aclnnMatmul`. If it fails, it uses the stable fallback:

```text
F16 activation -> F32
aclnnMm(F32 activation, FRACTAL_NZ weight) -> F32 result
F32 result -> F16 output
```

This preserves the F16/NZ fast path for normal prefill shapes while keeping the
known bad tiling shapes runnable.

#### 310P FlashAttention mask semantics

The 310P implementation uses two attention operators:

| Stage | Operator | Mask interface |
|---|---|---|
| decode, `q_len == 1` | `aclnnIncreFlashAttentionV4` | additive `pseShift` |
| prefill, `q_len > 1` | `aclnnPromptFlashAttentionV3` | bool `attenMask` |

The GGML F16 mask convention is `0 = keep` and `-inf = masked`.

For decode, a null mask is allowed. In that case no `pseShift` is built and
`nullptr` is passed to IFA. When an explicit mask exists, the F16 mask is
repeated across heads and passed as `pseShift`.

For prefill, PFA V3 on 310P requires a bool `attenMask` and does not accept the
additive `pseShift` interface. The implementation allocates a `K x K` bool
mask, fills it with `false`, and uses `false` as the keep value. If an explicit
F16 mask is present, its `q_len x K` prefix is cast to bool, where `0` becomes
`false` and `-inf` becomes `true`.

There is one important safety gate: 310P PFA prefill with a null mask has shown
incorrect numerical behavior for full attention. Therefore `supports_op`
rejects `q_len > 1 && mask == nullptr` on 310P. Those prefill ops fall back to
CPU softmax even though the runtime helper can construct an all-keep mask.
Decode ops with a null mask remain on NPU.

#### RoPE adaptation for Qwen3-VL

Qwen3-VL uses M-RoPE/IMROPE for text plus image inputs and vision RoPE inside
the CLIP graph. The old 310P assertion rejected the M-RoPE/IMROPE mode. It now
accepts:

```text
full-width RoPE || vision RoPE || M-RoPE/IMROPE
```

The Qwen3-VL `Qcur` and `Kcur` tensors are views produced by splitting QKV. On
310P, partial RoPE requires contiguous input, so these tensors are copied to
contiguous buffers before `ggml_rope_multi`. This small copy keeps vision RoPE
on NPU and avoids CPU fallback or graph construction failure.

#### Other operator and memory fixes

1. **Bilinear upscale**

   The CANN backend previously supported only `UpsampleNearest2d`. Qwen3-VL
   position-embedding interpolation uses bilinear scaling, which forced CPU
   fallback. The backend now dispatches:

   ```text
   bilinear -> aclnnUpsampleBilinear2d
   nearest  -> aclnnUpsampleNearest2d
   ```

   The `align_corners` flag is read from the GGML op parameters. This removes
   the last known unsupported op in the tested Qwen3-VL warmup.

2. **Dense-weight NZ candidate expansion**

   Dense weights no longer have to be explicitly F16 to become NZ-layout
   candidates. The remaining exclusions are still enforced:

   - `GGML_CANN_WEIGHT_NZ` must be enabled;
   - BF16 and W8A8 remain excluded;
   - tensors must be contiguous;
   - the first two dimensions must be 32-aligned;
   - expert weights remain restricted to F16.

3. **Pinned prefill causal mask**

   The paged prefill causal mask is filled on host memory and copied to device
   with `aclrtMemcpyAsync`. A normal pageable host allocation cannot be used in
   an ACL graph capture. The host mask is now allocated with
   `aclrtMallocHost`, and both the device and pinned host buffers are cached in
   the paged attention state. When the mask size changes, the stream is
   synchronized before the old buffers are freed.

   This makes the H2D copy capture-safe and prevents the source buffer from
   being freed before the async copy completes.

## vLLM Ascend comparison

### Qwen3-1.7B text, 10 requests

Historical best AISBench runs:

| Engine | TTFT avg | TPOT avg | E2EL avg | Aggregate output throughput |
|---|---:|---:|---:|---:|
| llama.cpp | 5816.5 ms | 69.5 ms | 148180.8 ms | 138.179 tok/s |
| vLLM Ascend | 5733.6 ms | 63.8 ms | 136398.0 ms | 149.691 tok/s |

The input lengths differ slightly in this historical run: llama.cpp averages
4104 tokens and vLLM averages 4317.5 tokens after tokenizer/chat-template
processing. The remaining gap is mainly stable-decode latency and host-side
work, not a missing PA kernel.

### Qwen3-VL-4B, single request

| Scenario | Engine | TTFT avg | TPOT avg | E2EL avg | Aggregate output throughput |
|---|---|---:|---:|---:|---:|
| 320x320 image + text | llama.cpp | 139.7 ms | 65.0 ms | 8388.9 ms | 15.258 tok/s |
| 320x320 image + text | vLLM Ascend | 145.8 ms | 59.2 ms | 7667.2 ms | 16.695 tok/s |
| 1280x720 image + text | llama.cpp | 416.6 ms | 83.0 ms | 10954.0 ms | 11.685 tok/s |
| 1280x720 image + text | vLLM Ascend | 356.8 ms | 78.4 ms | 10309.2 ms | 12.416 tok/s |

The llama.cpp VL small TTFT is already comparable. The main remaining VL gap is
decode latency, especially for the large image path.

## Negative results

The following experiments were tried and should not be treated as safe defaults:

| Experiment | Result |
|---|---|
| Direct F16 RMSNorm | Wrong output and lower prefill throughput. |
| Three-input AddRmsNorm + Mul fusion | Semantic drift in real workloads. |
| Ubatch 2048 | Exceeded the available compute buffer on the tested model. |
| `cubeMathType=3` | Not supported for F32 cube execution on 310P. |
| `cubeMathType=2` | No stable prefill gain. |
| 512 MB F16 input cache | No improvement over 256 MB. |
| Upstream ADD + RMS_NORM fusion | Within run-to-run noise. |
| `aclnnFusedMatmul` | No stable end-to-end gain after the required casts. |
| Temporary F16 matmul output plus F32 cast | No stable gain in the 10-request workload. |
| Pure chunk GDN state | Prefill is faster but has accuracy drift under non-zero initial state. |
| Early ACL graph replay without phase gating | Replayed, but TTFT and TPOT regressed on Qwen3.5-2B. |
| `aclnnMatmulWeightNz` | Unsupported on the tested Ascend 310P build. |

## Current next steps

1. Keep the PA output in F16 and remove the PA output ViewCopy/Cast round trip.
2. Stabilize the decode activation domain as F16 where the graph allows it.
3. Reduce MatMul boundary transdata by reusing F16/NZ activations.
4. Fuse RoPE, QK norm, residual norm, and SwiGLU where the backend provides
   composite kernels.
5. Move sampling, logits handling, and next-batch preparation off the critical
   decode path.
6. Add chunked-prefill scheduling so long prompts do not repeatedly delay an
   already running decode batch.
7. Continue using the AISBench regression cases for every performance change.

## Related files

- `tools/server/tests/model-integration/README.md`
- `tools/server/tests/model-integration/scripts/run_all.py`
- `docs/backend/CANN-Hybrid-PagedAttention-Design.md`
- `docs/backend/CANN-Hybrid-PagedAttention-PrefixCache-Design.md`
- `docs/design/Paged-Server-Integration-Design.md`
