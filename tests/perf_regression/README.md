# Performance regression cases

This directory contains three repeatable AISBench cases for llama-server and a vLLM Ascend baseline:

| scenario | model | workload | target |
|---|---|---|---|
| text_10c | Qwen3 1.7B | 10 concurrency, about 4096 input tokens, 2048 output tokens | TTFT <= 500 ms, TPOT <= 14 ms |
| vl_small | Qwen3 VL 4B | one 320x320 JPEG + text, 128 output tokens | E2E <= 30 ms |
| vl_large | Qwen3 VL 4B | one 1280x720 JPEG + text, 128 output tokens | E2E <= 1000 ms |

The llama.cpp case enables the CANN fast paths: ACL graph, FA V5, F16 activation cache, weight NZ, and the available GDN prefill/decode fused paths. GDN operators are model-dependent; they are enabled by environment variables and are skipped automatically when a model has no GDN layers.

## Remote layout

Copy this directory to:

```bash
/home/z00575061/llama.cpp-develop-qwen3-20261102/tests/perf_regression
```

Run from the CANN container. The default NPU is device 7.

## Build and prepare

```bash
cd /home/z00575061/perf_regression
chmod +x scripts/*.sh
python scripts/prepare_assets.py
python scripts/validate_env.py
LLAMA_SOURCE=/home/z00575061/llama.cpp-develop-qwen3-20261102 \
  scripts/build_llama.sh
```

`validate_env.py` fails if the llama-server build was not rebuilt with `USE_ACL_GRAPH=ON`.

## Start one service

llama.cpp:

```bash
LLAMA_MODEL=/home/z00575061/models/Qwen3-1.7B-F16.gguf \
PERF_LLAMA_CTX=65536 \
PERF_LLAMA_PARALLEL=10 \
scripts/run_llama_server.sh
```

vLLM baseline:

```bash
VLLM_MODEL=/home/c30081098/models/Qwen3-1.7B \
scripts/run_vllm_server.sh
```

For Qwen3 VL, change `LLAMA_MODEL` to `/home/common/model/qwen3/Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-F16.gguf`. The server uses `--mmproj-auto` because the GGUF contains the projector.

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

Raw AISBench outputs are under `results/<engine>/<scenario>/<timestamp>`. Aggregated data is written to `results/summary.json`, `results/summary.md`, and `results/comparison.md`.

For the 10-concurrency text case, the current CANN ACL graph path can abort with
`Insufficient_Event_Resources` during long capture sessions. Use
`PERF_LLAMA_ACL_GRAPH=0` for that case until the graph event lifecycle is fixed.
This remains a compile-time ACL-graph build, but the text case runs eager.

If starting AISBench manually, prepend the config directory to `PYTHONPATH`:

```bash
PYTHONPATH=$PWD/configs ais_bench ...
```
