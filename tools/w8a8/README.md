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

## Direct ModelSlim import

`convert_modelslim.py` imports a ModelSlim `W8A8_DYNAMIC` checkpoint directly
without dequantizing and requantizing it:

```bash
python tools/w8a8/convert_modelslim.py \
  /path/to/Qwen3.5-9B-W8A8-310P \
  Qwen3.5-9B-W8A8-310P.gguf
```

Use `--dry-run` first to validate all W8A8 triples and print the source-to-GGUF
tensor mapping. Use `--mapping-out mapping.json` to save the mapping report.

The importer:

- supports `Qwen3MoeForCausalLM` text-only checkpoints;
- supports `W8A8_DYNAMIC` expert triples and, with
  `--allow-static-w8a8`, static `W8A8` attention weights whose BF16 scales
  are converted to FP32 before packing;
- preserves the original INT8 weight payload;
- preserves the FP32 scale subject to the existing FP16 GGUF scale slot;
- rejects nonzero `weight_offset` values;
- leaves non-W8A8 tensors in F16/BF16/F32 according to the existing converter;
- reuses the existing Qwen model mapping, tokenizer, MTP, and metadata logic
  from `convert_hf_to_gguf.py`.

For the static `W8A8` variant, `input_scale`, `input_offset`, `quant_bias`,
and `deq_scale` are deliberately omitted because the llama.cpp W8A8 runtime
computes dynamic activation quantization. The mapping report records every
omitted tensor, and the flag must be supplied explicitly.

`tools/w8a8/inspect_modelslim.py` inventories a checkpoint before conversion.

### MoE checkpoints

For `Qwen3_5MoeForConditionalGeneration`, the importer detects separate
per-expert `gate_proj`, `up_proj`, and `down_proj` modules and stacks them into
the native 3D GGUF expert tensors. It also accepts fused `gate_up_proj` and
`down_proj` tensors. The runtime shapes are:

```text
gate/up:  GGML ne [K,N,E]   public raw [E,N,(K/32)*34]
down:     GGML ne [K,N,E]   public raw [E,N,(K/32)*34]
scales:   [E,N]
```

The converter streams tensors through a temporary GGUF file by default. Use
`--no-temp-file` only when there is enough host RAM to retain the whole output.
A vision-language MoE model can emit the projector separately:

```bash
python tools/w8a8/convert_modelslim.py \
  /path/to/Qwen3.5-35B-W8A8-310P \
  Qwen3.5-35B-W8A8-310P.gguf \
  --mapping-out mapping.json \
  --mmproj-output mmproj-Qwen3.5-35B-f16.gguf
```

## Verify a converted model

```bash
python tools/w8a8/verify_modelslim_gguf.py \
  /path/to/Qwen3.5-9B-W8A8-310P \
  Qwen3.5-9B-W8A8-310P.gguf \
  mapping.json \
  --samples 8
```

The verifier checks GGUF metadata and compares sampled INT8 payloads and
FP16 block scales against the original ModelSlim weights.

## Vision projector sidecar

```bash
python tools/w8a8/convert_modelslim.py \
  /path/to/Qwen3.5-9B-W8A8-310P \
  text.gguf \
  --mmproj-only \
  --mmproj-output mmproj-Qwen3.5-9B-f16.gguf
```

The main text GGUF preserves the language model and MTP block. Vision and
projector tensors are written separately in `clip` architecture format.
