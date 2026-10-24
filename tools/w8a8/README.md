# W8A8 offline adapter

`convert_w8a8.py` converts a simple explicit symmetric W8A8 artifact to a
tensor-only GGUF file. The input `.npz` must contain:

- `weight`: INT8, shape `[N,K]`, output-channel major;
- `weight_scale`: FP16 or FP32, shape `[N]`;
- optional `weight_offset`: only all-zero values are accepted.

`K` must be divisible by 32. The output uses the fork-local `W8A8` GGML type:
each 32-value block contains 32 INT8 payloads followed by one FP16 scale, and
the scale is repeated across every block in an output-channel row.

Example:

```bash
python tools/w8a8/convert_w8a8.py input.npz output.gguf --tensor-name weight
```
