# Qwen3-VL DFlash acceptance diagnostics

These are manual experiment tools, separate from the pytest/CTest regression suite. They send real serial HTTP requests to `llama-server`, compare converted weights, capture a first draft block, and replay that block with eager PyTorch arithmetic. They do not download models or include datasets, generated answers, tensor dumps, model files, or compiled probes.

The tools retain the logic of the 2026-10-29/30 Ascend 310P3 experiments. Paths, devices and output directories are arguments; the shared request path is imported directly rather than loaded with string replacement and `exec`.

## Dependencies and scope

- Python with `requests` for HTTP runs.
- NumPy, safetensors and an existing PyTorch installation for weight/reference checks.
- The repository's `gguf-py` package is loaded from the checkout.
- Linux, a C++17 compiler and the exact runtime headers/libraries for `capture.cpp`.
- A target GGUF, matching mmproj, a block-size-16 DFlash GGUF, and local images.

The numerical reference and capture probe support the tested five-layer Qwen3-VL contract: hidden size 2560, head dimension 128, 32 Q heads, 8 KV heads, five target feature taps, MRoPE sections `[24,20,20]`, interleaved frequencies, theta 5000000, and RMS epsilon 1e-6. The reference rejects other declared configurations. The capture probe is single-sequence and reads four position rows. It is not a general-purpose model or multi-sequence tracer.

Run the commands below from the repository root. With copied scripts outside a checkout, set `LLAMA_CPP_ROOT` to the checkout for the two GGUF readers and pass `--repo-root` to HTTP tools. Use a new output directory for every run.

## Paired single-concurrency benchmark

The JSONL schema is one object per line:

```json
{"question_id": 1, "prompt": "What does the sign say?", "image_files": ["images/sign.jpg"], "answer": "example"}
```

Relative image paths are resolved against the dataset directory. `answer` is kept as evidence and is not sent to the model. Question IDs default to row indices.

```bash
python tools/server/tests/diagnostics/dflash_vl/benchmark.py \
  --server /path/to/build/bin/llama-server \
  --target /models/target-f16.gguf --mmproj /models/mmproj-f16.gguf \
  --draft /models/draft-mrope-f16.gguf --dataset /datasets/textvqa.jsonl \
  --output /results/paired-k15 --visible-devices 6 --device CANN0 \
  --samples 200 --warmups 5 --draft-max 15 --max-tokens 64
```

Defaults match the measured protocol: context 4096, batch/ubatch 128, four CPU threads, FA off, reasoning off, F16 K/V, image budget min=max=1024, greedy sampling, seed 1234, selection seed 20261029, one slot, no prompt cache, CPU draft sampling. The image patch grid may round the requested image-token budget.

`--visible-devices` sets `ASCEND_RT_VISIBLE_DEVICES`; `--device` is the resulting backend name. The tools only stop child servers they started. The port is loopback and selected dynamically. Performance runs clear preload/debug timing variables.

Use `--prepare-only` to validate input files and save the fixed sample selection without starting a server. `--monitor-npu` optionally records `npu-smi` snapshots. The portable benchmark starts a fresh baseline followed by a fresh DFlash server; it does not silently reuse an earlier baseline.

Outputs include the plan, warmups, exact server commands, per-request JSONL, paired comparisons, summary and owned-process cleanup records. Warmups and model loading are excluded from the measured request phase. Image reads/base64 encoding are done before timing. SSE TTFT is the first nonempty text fragment.

Metrics:

- Acceptance fraction: `sum(accepted) / sum(drafted)`.
- Accepted draft tokens per verification: `sum(accepted) / sum(rounds)`.
- Native `mean_acc_len`: `1 + sum(accepted) / sum(rounds)`.
- Decode throughput: `sum(max(predicted_n - 1, 0)) / sum(predicted_ms / 1000)`. The first token comes from the final prompt logits in this server revision.
- Client latency: POST start through SSE `[DONE]`.

Text and stop-reason mismatches produce a failed comparison while retaining raw responses. Empty generated-ID arrays are reported as unavailable, never as a successful token-ID comparison. A clean text comparison is not a TextVQA accuracy score or proof that every internal tensor matches.

## Acceptance controls

```bash
python tools/server/tests/diagnostics/dflash_vl/controls.py \
  --server /path/to/build/bin/llama-server \
  --target /models/target-f16.gguf --mmproj /models/mmproj-f16.gguf \
  --draft /models/draft-mrope-f16.gguf --dataset /datasets/textvqa.jsonl \
  --output /results/controls --visible-devices 6 --samples 200 --warmups 0 \
  --variants batch128 batch1024 legacy1d image256 \
  --legacy-draft /models/draft-temporal-f16.gguf
```

The first three selected images are reused across variants. `batch128` also sends three fixed text prompts (numbers, explanation, code), with output limit 128. Images use `--max-tokens` (default 64). These short controls are acceptance probes, not a cross-task quality ranking or a warmed performance benchmark.

`legacy1d` requires a separate GGUF whose weights have been checked against the current draft. `image256` changes image representations as well as token count.

## Converted-weight check

```bash
python tools/server/tests/diagnostics/dflash_vl/check_weights.py \
  --hf-dir /models/draft-hf --gguf /models/draft-mrope-f16.gguf \
  --output /results/weight-parity.json
```

This checks a single `model.safetensors` checkpoint. It maps names with gguf-py, casts each HF tensor to the GGUF tensor's storage dtype, then checks shape, values, finiteness and tensor counts. A mismatch returns nonzero. It is not a general quantized-weight validator or a test of training quality.

## First-block capture and eager reference

Build the probe against the same headers and actual shared library used by the server. Use the real versioned library if the unversioned linker alias is absent:

```bash
g++ -std=c++17 -O2 -shared -fPIC \
  -Iinclude -Iggml/include \
  tools/server/tests/diagnostics/dflash_vl/capture.cpp \
  /path/to/build/bin/libllama.so.0.4.0 \
  -Wl,-rpath,/path/to/build/bin -ldl -o /results/capture.so
```

Run `controls.py` with the same model/dataset arguments, a new output directory, `--variants capture3d capture1d`, `--capture-library /results/capture.so` and `--legacy-draft /models/draft-temporal-f16.gguf`.

The controller arms capture only after server startup. It records the first image request's target feature batches, four position rows, anchor/mask IDs and raw draft logits, then stops capture after the first noise block. Capture requests have an output limit of 32. The probe does not directly dump each layer's K/V cache. It synchronizes and writes tensors; capture timings are not performance results.

```bash
python tools/server/tests/diagnostics/dflash_vl/torch_reference.py \
  --capture /results/capture-run/capture3d/capture \
  --hf-dir /models/draft-hf --target-gguf /models/target-f16.gguf \
  --output /results/torch-reference.json
```

The reference uses the captured target features, original draft checkpoint, target GGUF token embeddings/output head, and declared MRoPE settings. It recomputes all five draft layers in eager F32, with F16 K/V storage emulation. It reports relative L2, cosine, argmax ties and a comparison excluding spatially distinct rows. It is an independent arithmetic check of the declared standard DFlash contract, not the unavailable original training forward. It does not independently validate target feature extraction or the exact target checkpoint used for training.

## Recorded experiment results (before CLI packaging)

On 2026-10-29, 200 paired Qwen3-VL-4B F16 requests on one Ascend 310P3, FA off, output limit 64, returned identical text and stop reasons. The baseline was reused when rerunning DFlash at k=15. Mean end-to-end latency was 23.71 s baseline and 26.28 s DFlash; decode throughput was 8.99 and 6.65 token/s. Accepted draft tokens were 6977 over 5050 verification steps: 1.38158 draft tokens/step, or native mean acceptance length 2.38158 including the bonus token.

On 2026-10-30, 58/58 converted tensors matched. The three text controls accepted 10.55, 1.82 and 3.88 draft tokens/step. Three image cases had identical counters at batch/ubatch 128 and 1024, and with temporal-only vs three-axis draft metadata. For one real image's first block, relative logits L2 against the eager reference was 0.000503138 and mean cosine 0.9999991655. Fourteen of fifteen strict argmax IDs matched; the remaining choice was tied in native logits, so all fifteen reference choices belonged to the native maximum-score sets.

Those results support the tested conversion and runtime paths, not complete training/deployment parity or a universal acceleration claim. The exact training forward, target-weight revision and training input template remain unverified. Logs, raw captures and private dataset/model paths stay in external run archives.
