# W8A8 MoE FRACTAL_NZ path on Ascend 310P

This branch adds an opt-in FRACTAL_NZ weight path for W8A8 routed experts:

```text
GGML_CANN_MOE_WEIGHT_NZ=1
```

The existing ND grouped-matmul path remains the default.

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
- The path remains opt-in.

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

Real model, Qwen3.5-35B-A3B W8A8, `llama-bench`, two 310P devices,
`-sm layer -ts 0.55,0.45`:

| Metric | ND | NZ | Speedup |
|---|---:|---:|---:|
| pp512 tok/s | 199.64 | 340.60 | 1.71x |
| tg128 tok/s | 4.57 | 8.80 | 1.93x |

## Accuracy

Fixed 8192-byte sample, context 512, three chunks:

| Path | PPL |
|---|---:|
| ND | 1.0317 +/- 0.00844 |
| NZ | 1.0317 +/- 0.00844 |

The one-time NZ conversion added about 19 seconds to model load in this test
(`27.59s` to tokenization for ND versus `46.94s` for NZ).
