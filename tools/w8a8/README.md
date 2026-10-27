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

## Complete GGUF model quantization

The tensor-only adapter above is useful for explicit weight artifacts. For a
complete model GGUF, use `llama-quantize` with the fork-local `W8A8` file type:

```bash
llama-quantize input-f16.gguf output-w8a8.gguf W8A8
```

By default, dense layer weights are converted to W8A8, token embeddings remain
F16 for `GET_ROWS`, and the output projection follows the quantizer's normal
mixed-type policy. To force the output projection to W8A8 as well:

```bash
llama-quantize --output-tensor-type w8a8 input-f16.gguf output-w8a8.gguf W8A8
```

The A8 runtime path is enabled by default. Set `GGML_CANN_W8A8_A8=0` to use
the W8A16 fallback.
