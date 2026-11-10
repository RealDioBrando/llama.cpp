# Model integration tests

This directory contains three repeatable AISBench cases for llama-server and a vLLM Ascend baseline:

| scenario | model | workload | target |
|---|---|---|---|
| text_10c | Qwen3 1.7B | 10 concurrency, about 4096 input tokens, 2048 output tokens | TTFT <= 500 ms, TPOT <= 14 ms |
| vl_small | Qwen3 VL 4B | one 320x320 JPEG + text, 128 output tokens | E2E <= 30 ms |
| vl_large | Qwen3 VL 4B | one 1280x720 JPEG + text, 128 output tokens | E2E <= 1000 ms |

The llama.cpp case enables the CANN fast paths: ACL graph, FA V5, F16 activation cache, weight NZ, and the available GDN prefill/decode fused paths. GDN operators are model-dependent; they are enabled by environment variables and are skipped automatically when a model has no GDN layers.

Accuracy cases use AISBench. The text case uses GSM8K and the VL case uses
TextVQA val images from OpenImages. `run_all.py` uses the first 30 unique
TextVQA images by default. To use the full dataset, pass `--accuracy-full`
or set `--accuracy-num-prompts 0`.

TextVQA generation is capped at 16 tokens. The VQA evaluator accepts a
prediction when the normalized reference answer is its prefix, so a short
answer followed by descriptive text is not counted as a mismatch.

The VL cases use `--no-mmproj-offload` by default because the CANN GPU CLIP
RoPE path aborts on the tested build. Set `PERF_LLAMA_MMPROJ_OFFLOAD=1` to
restore GPU offloading when that path is fixed.

## Remote layout

The scripts derive the repository root from their own location:

```bash
llama.cpp/tools/server/tests/model-integration
```

Run from the CANN container. The default NPU is device 7, but every model path is supplied through environment variables.

## Build and prepare

```bash
chmod +x scripts/*.sh
python scripts/prepare_assets.py
python scripts/prepare_accuracy_assets.py
python scripts/validate_env.py
LLAMA_TEXT_MODEL=/path/to/Qwen3-1.7B-F16.gguf \
LLAMA_VL_MODEL=/path/to/Qwen3VL-4B-Instruct-F16.gguf \
  python scripts/validate_env.py
scripts/build_llama.sh
```

`validate_env.py` fails if the llama-server build was not rebuilt with `USE_ACL_GRAPH=ON`.

## Start one service

llama.cpp:

```bash
LLAMA_TEXT_MODEL=/path/to/Qwen3-1.7B-F16.gguf \
PERF_LLAMA_CTX=65536 \
PERF_LLAMA_PARALLEL=10 \
scripts/run_llama_server.sh
```

vLLM baseline:

```bash
VLLM_TEXT_MODEL=/path/to/Qwen3-1.7B \
scripts/run_vllm_server.sh
```

For Qwen3 VL, set:

```bash
export LLAMA_VL_MODEL=/path/to/Qwen3VL-4B-Instruct-F16.gguf
export VLLM_VL_MODEL=/path/to/Qwen3-VL-4B-Instruct
```

The server uses `--mmproj-auto` because the GGUF contains the projector. If the projector is a separate GGUF, set `LLAMA_VL_MMPROJ=/path/to/Qwen3VL-4B-Instruct-mmproj-f16.gguf`.

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
python scripts/run_all.py --engine llama --scenario text_10c --accuracy-mode only
python scripts/run_all.py --engine llama --scenario vl_small --accuracy-mode only
```

Run the full accuracy datasets:

```bash
python scripts/run_all.py --engine llama --scenario all --accuracy-mode only --accuracy-full
```

Raw AISBench outputs are under `results/<engine>/<scenario>/<timestamp>`. Aggregated data is written to `results/summary.json`, `results/summary.md`, and `results/comparison.md`.

For the 10-concurrency text case, the current CANN ACL graph path can abort with
`Insufficient_Event_Resources` during long capture sessions. Use
`PERF_LLAMA_ACL_GRAPH=0` for that case until the graph event lifecycle is fixed.
This remains a compile-time ACL-graph build, but the text case runs eager.

If starting AISBench manually, prepend the config directory to `PYTHONPATH`:

```bash
PYTHONPATH=$PWD/configs ais_bench ...
```
