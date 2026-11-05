# W8A8 MoE FRACTAL_NZ path on Ascend 310P

This branch enables the FRACTAL_NZ weight path by default for W8A8 routed
experts on Ascend 310P. The ND path remains available for comparison and
debugging:

```text
GGML_CANN_MOE_WEIGHT_NZ=0
```

A non-310P build keeps ND as the default.

## Layout

For an expert tensor `[E, N, K]`, the 310P INT8 FRACTAL_NZ payload is stored as:

```text
[E, K/32, N/16, 16, 32]
```

The public W8A8 blocks are unpacked during upload, converted to the NZ payload,
and described to ACLNN as an INT8 FRACTAL_NZ tensor with the logical view
`[E, N, K]`. The host-side conversion uses 32-byte block copies. `get_tensor`
performs the inverse conversion so callers always observe the public 34-byte
W8A8 block representation.

The conversion was checked byte-for-byte against:

```text
torch_npu.npu_format_cast(tensor, torch_npu.Format.FRACTAL_NZ)
```

for a representative `[16, 64, 64]` expert tensor.

## Constraints

- Ascend 310P INT8 FRACTAL_NZ.
- Expert weights only; dense W8A8 uses its existing transformed path.
- `K % 32 == 0`.
- `N % 16 == 0`.
- The path is default-on for 310P and can be disabled with `GGML_CANN_MOE_WEIGHT_NZ=0`.

## Correctness

Device: Ascend 310P3, CANN 9.1.0, driver 25.5.0.

| Test set | ND | NZ |
|---|---:|---:|
| `MUL_MAT_ID`, W8A8, F16 and F32 activations | 13/13 | 13/13 |
| `MUL_MAT`, W8A8 dense regression | 10/10 | 10/10 |

The tests include Qwen3-class routed-expert shapes, multi-token routing, empty
expert groups, and both gate/up and down projections.

## Performance

Backend microbenchmark, Qwen3-30B-A3B-sized routed projections:

| Test | ND us/run | NZ us/run | Speedup |
|---|---:|---:|---:|
| gate/up, 1 token | 1639.43 | 555.87 | 2.95x |
| down, 1 token | 1625.53 | 573.84 | 2.83x |
| gate/up, 32 tokens | 8507.30 | 1210.75 | 7.03x |
| down, 32 tokens | 6718.73 | 1606.79 | 4.18x |

Real model, Qwen3.5-35B-A3B W8A8, `llama-bench`, CANN1/CANN4,
`-sm layer -ts 0.60,0.40`:

| Metric | ND | NZ | Speedup |
|---|---:|---:|---:|
| pp512 tok/s | 192.98 | 335.66 | 1.74x |
| tg128 tok/s | 4.53 | 9.04 | 2.00x |

## Accuracy

Fixed 8192-byte sample, context 512, three chunks:

| Path | PPL |
|---|---:|
| ND | 1.0317 +/- 0.00844 |
| NZ | 1.0317 +/- 0.00844 |

AISBench GSM8K, first 20 cases, temperature 0, thinking disabled:

| Output cap | ND | NZ |
|---|---:|---:|
| 512 tokens | 19/20 | 18/20 |
| Cases 8-12 rerun at 768 tokens | 4/5 | 4/5 |

At 512 tokens, NZ case 8 stopped at the output limit before emitting an answer.
Case 12 failed on both paths with the same incorrect answer (`12` instead of
`13`). The 768-token rerun completed case 8 correctly on both paths, making the
effective result 19/20 for both paths with a sufficient output budget.

The model used for these runs has 256 routed experts, top-8 routing, hidden
size 2048, and MoE intermediate size 512. The microbenchmark cases use a second
shape with 128 experts, top-8 routing, hidden size 2048, and MoE intermediate
size 768.

The one-time NZ conversion added about 19 seconds to model load in this test
(`27.59s` to tokenization for ND versus `46.94s` for NZ).

## ACL graph mode

An `USE_ACL_GRAPH=ON` build also passes the W8A8 backend matrix:

| Test set | ND | NZ |
|---|---:|---:|
| `MUL_MAT_ID`, W8A8 | 13/13 | 13/13 |
| `MUL_MAT`, W8A8 dense | 10/10 | 10/10 |

The NZ grouped-matmul workspace is not safe to replay from the CANN 9.1 graph
cache. A long server decode reproduced an AICore failure in
`DynamicQuant_fp16_1_high_performance_11000` with an invalid GM address. The
backend now detects W8A8 NZ expert nodes and automatically uses eager execution
for that graph, while an ACL graph ON build remains usable elsewhere.

After adding the fallback:

| Test | Result |
|---|---:|
| AISBench GSM8K, first 10 cases, 768 output tokens | 10/10 |
| `pp512` | 333.62 tok/s |
| `tg128` | 9.04 tok/s |

Matched `USE_ACL_GRAPH=ON` results after the fallback were `197.74 / 4.46`
tok/s for ND and `333.62 / 9.04` tok/s for NZ (`pp512 / tg128`). The NZ path
therefore remains about 1.69x faster at prefill and 2.03x faster at decode,
while avoiding the unstable graph-replay path.

Default selection was also verified without setting
`GGML_CANN_MOE_WEIGHT_NZ`:

| Configuration | Result |
|---|---:|
| ACL graph OFF, default NZ, W8A8 `MUL_MAT_ID` | 13/13 |
| ACL graph ON, default NZ, AISBench GSM8K 10 cases | 10/10 |
| ACL graph ON, default NZ, `pp512` / `tg128` | 336.28 / 8.82 tok/s |
