# Model integration tests

This directory contains three repeatable AISBench cases for llama-server and a vLLM Ascend baseline:

| scenario | model | workload | target |
|---|---|---|---|
| text_10c | Qwen3 1.7B | 10 concurrency, 10 GSM8K questions padded to about 4096 input tokens, 2048 output tokens | TTFT <= 500 ms, TPOT <= 14 ms |
| vl_small | Qwen3 VL 4B | one 320x320 JPEG + text, 128 output tokens | E2E <= 30 ms |
| vl_large | Qwen3 VL 4B | one 1280x720 JPEG + text, 128 output tokens | E2E <= 1000 ms |

The llama.cpp case enables the CANN fast paths: ACL graph, FA V5, F16 activation cache, weight NZ, and the available GDN prefill/decode fused paths. GDN operators are model-dependent; they are enabled by `config.json` and are skipped automatically when a model has no GDN layers.

Accuracy cases use AISBench. The text case uses GSM8K and the VL case uses
TextVQA val images from OpenImages. `run_all.py` uses the first 30 unique
TextVQA images by default. To use the full dataset, pass `--accuracy-full`
or set `--accuracy-num-prompts 0`.

TextVQA generation is capped at 16 tokens. The VQA evaluator accepts a
prediction when the normalized reference answer is its prefix, so a short
answer followed by descriptive text is not counted as a mismatch.

The VL cases enable mmproj offload by default. Set
`PERF_LLAMA_MMPROJ_OFFLOAD=0` to keep the CLIP encoder on CPU when
debugging CANN-specific regressions.

## Configuration

`config.json` is the single default source for runtime paths, models, datasets,
scenarios, and server settings. It is organized as:

| section | contents |
|---|---|
| `runtime` | NPU IDs, vLLM container, build paths/options, CANN/ATB env scripts, AISBench paths |
| `models` | llama.cpp GGUF/mmproj paths, vLLM model directories, and tokenizer/HF directories |
| `datasets` | dataset type, source path, subset size, tokenizer model, and image metadata. AISBench built-in datasets use `aisbench://`; local files use paths relative to this directory and are resolved to absolute paths. |
| `scenarios` | model/dataset binding, AISBench workload, targets, and model-config names |
| `servers` | llama.cpp and vLLM defaults plus per-scenario overrides and backend feature flags |

Precedence is command-line option, then `config.json`. All entry scripts accept
`--config`; if it is omitted, they use the adjacent `config.json`. To use an
alternate configuration file, pass it on the command line:

```bash
python scripts/run_all.py --config /path/to/config.json --engine llama --scenario all
bash scripts/run_llama_server.sh text_10c --config /path/to/config.json
bash scripts/run_vllm_server.sh text_10c --config /path/to/config.json
bash scripts/build_llama.sh --config /path/to/config.json
```

New server tuning knobs should be added under `servers.<engine>.defaults` or
`servers.<engine>.scenarios.<name>`. A scenario value overrides the same field
in the engine defaults.

## Perf dataset overrides

`run_all.py` can point the perf workload at custom data without editing
`config.json`:

```bash
# custom VL image/question jsonl (same format as assets/image_320x320_questions.jsonl)
python scripts/run_all.py --engine llama --scenario vl_small --bench-mode perf \
  --perf-dataset-path /path/to/images.jsonl

# custom text perf data (same format as the gsm8k fixed-token dataset)
python scripts/run_all.py --engine llama --scenario text_10c --bench-mode perf \
  --perf-dataset-path /path/to/questions.jsonl
```

`--perf-dataset <key>` selects a different dataset entry from the config
`datasets` section for the selected scenarios. `--perf-dataset-path` takes
precedence over the dataset path in the config.

## Remote layout

The scripts derive the repository root from their own location:

```bash
llama.cpp/tools/server/tests/model-integration
```

Run from the CANN container. The default NPU is device 5. Model paths have
`/home/common` defaults from `config.json` and can be overridden with CLI
options or the previous environment variables.

## Build and prepare

```bash
chmod +x scripts/*.sh
python scripts/prepare_assets.py
python scripts/prepare_accuracy_assets.py
python scripts/validate_env.py
scripts/build_llama.sh
```

`validate_env.py` fails if the llama-server build was not rebuilt with `USE_ACL_GRAPH=ON`.

## Start one service

llama.cpp:

```bash
PERF_LLAMA_CTX=65536 \
PERF_LLAMA_PARALLEL=10 \
scripts/run_llama_server.sh text_10c --model /path/to/Qwen3-1.7B-F16.gguf
```

vLLM baseline:

```bash
scripts/run_vllm_server.sh text_10c --model /path/to/Qwen3-1.7B
```

For Qwen3 VL, pass:

```bash
scripts/run_llama_server.sh vl_small \
  --model /path/to/Qwen3VL-4B-Instruct-F16.gguf \
  --mmproj /path/to/Qwen3VL-4B-Instruct-mmproj-f16.gguf
scripts/run_vllm_server.sh vl_small --model /path/to/Qwen3-VL-4B-Instruct
```

The server uses the separate mmproj by default. `run_all.py` accepts
`--llama-text-model`, `--llama-vl-model`, `--llama-vl-mmproj`,
`--qwen3-1-7b-path`, `--qwen3-vl-4b-path`, `--vllm-text-model`, and
`--vllm-vl-model`.

The fixed-token dataset uses the `tokenizer` path of `qwen3_1_7b` from
`config.json`. `QWEN3_1_7B_PATH` or the `--qwen3-1-7b-path` option overrides it
for one run.

## Run cases

All llama.cpp cases:

```bash
python scripts/run_all.py --engine llama --scenario all
```

All vLLM cases and save them as the baseline:

```bash
python scripts/run_all.py --engine vllm --scenario all --save-baseline
```

One case:

```bash
python scripts/run_all.py --engine llama --scenario text_10c
python scripts/run_all.py --engine llama --scenario vl_small
python scripts/run_all.py --engine llama --scenario vl_large
```

Generate the comparison:

```bash
python scripts/compare_report.py
```

Run accuracy only:

```bash
python scripts/run_all.py --engine llama --scenario text_10c --bench-mode accuracy
python scripts/run_all.py --engine llama --scenario vl_small --bench-mode accuracy
```

Run the full accuracy datasets:

```bash
python scripts/run_all.py --engine llama --scenario all --bench-mode accuracy --accuracy-full
```

Raw AISBench outputs are under `results/<engine>/<scenario>/<timestamp>`. Aggregated data is written to `results/summary.json`, `results/summary.md`, and `results/comparison.md`.

For the 10-concurrency text case, the current CANN ACL graph path can abort with
`Insufficient_Event_Resources` during long capture sessions. Use
`PERF_LLAMA_ACL_GRAPH=0` for that case until the graph event lifecycle is fixed.
This remains a compile-time ACL-graph build, but the text case runs eager.

`run_all.py` reads AISBench and CANN paths from
`config.json -> runtime.ais_bench` and `config.json -> runtime.cann`.
Environment variables `AIS_BENCH_BIN`, `AIS_BENCH_HOME`, or `CANN_SET_ENV`
override those values. If starting AISBench manually, prepend the config
directory to `PYTHONPATH`:

```bash
PYTHONPATH=$PWD/configs ais_bench ...
```
