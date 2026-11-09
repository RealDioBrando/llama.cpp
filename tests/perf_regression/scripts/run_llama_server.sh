#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_llama_server.sh [text_10c|vl_small|vl_large]
# Default: text_10c
SCENARIO=${1:-text_10c}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${PERF_REGRESSION_ROOT:-$(dirname "${SCRIPT_DIR}")}
BUILD_DIR=${LLAMA_BUILD_DIR:-/home/z00575061/llama.cpp-paged-integration-latest/build-cann-aclgraph}

# Source CANN environment and ATB libs so the server can find libascendcl etc.
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export LD_LIBRARY_PATH=/home/z00575061/atb_home/lib:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}

# Per-scenario defaults (overridable by env vars)
case "${SCENARIO}" in
  text_10c)
    DEFAULT_MODEL=/home/common/model/qwen3/Qwen3-1.7B-F16.gguf
    DEFAULT_MMPROJ=""
    DEFAULT_PORT=18112
    DEFAULT_CTX=8192
    DEFAULT_PARALLEL=10
    DEFAULT_ACT_TYPE=f16
    ;;
  vl_small|vl_large)
    DEFAULT_MODEL=/home/common/model/qwen3/Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-F16.gguf
    DEFAULT_MMPROJ=/home/common/model/qwen3/Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-mmproj-f16.gguf
    DEFAULT_PORT=18113
    DEFAULT_CTX=32768
    DEFAULT_PARALLEL=4
    DEFAULT_ACT_TYPE=f16
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    echo "Usage: bash run_llama_server.sh [text_10c|vl_small|vl_large]" >&2
    exit 1
    ;;
esac

MODEL=${LLAMA_MODEL:-${DEFAULT_MODEL}}
PORT=${PERF_SERVER_PORT:-${DEFAULT_PORT}}
CTX=${PERF_LLAMA_CTX:-${DEFAULT_CTX}}
PARALLEL=${PERF_LLAMA_PARALLEL:-${DEFAULT_PARALLEL}}
THREADS=${PERF_LLAMA_THREADS:-8}
THREADS_BATCH=${PERF_LLAMA_THREADS_BATCH:-8}
MMPROJ=${LLAMA_MMPROJ:-${DEFAULT_MMPROJ}}
MMPROJ_ARGS=(--mmproj-auto)
if [[ -n "${MMPROJ}" ]]; then
  MMPROJ_ARGS=(--mmproj "${MMPROJ}")
fi

export ASCEND_RT_VISIBLE_DEVICES=${PERF_NPU_ID:-7}
export GGML_CANN_ACL_GRAPH=${PERF_LLAMA_ACL_GRAPH:-1}
export GGML_CANN_PREFILL_USE_GRAPH=${PERF_LLAMA_PREFILL_GRAPH:-0}
export GGML_CANN_FA_V5=1
export GGML_CANN_WEIGHT_NZ=1
export GGML_CANN_GDN_PROJ=1
export GGML_CANN_GDN_SMALL=1
export GGML_CANN_GDN_PREFILL_CHUNK=1

mkdir -p "${ROOT}/logs" "${ROOT}/run"
LOG=${ROOT}/logs/llama-server-${PORT}.log

ACT_TYPE=${PERF_LLAMA_ACT_TYPE:-${DEFAULT_ACT_TYPE}}
ACT_ARGS=()
if [[ "${ACT_TYPE}" != "none" ]]; then
  ACT_ARGS=(-at "${ACT_TYPE}")
fi

"${BUILD_DIR}/bin/llama-server" \
  -m "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -c "${CTX}" \
  -np "${PARALLEL}" \
  -t "${THREADS}" \
  -tb "${THREADS_BATCH}" \
  -ngl 99 \
  -sm none \
  -mg 0 \
  --kv-paged \
  --cache-ram 0 \
  -kvbls 128 \
  -b 2048 \
  -ub 2048 \
  "${ACT_ARGS[@]}" \
  --metrics \
  "${MMPROJ_ARGS[@]}" \
  2>&1 | tee "${LOG}" &
PID=$!
echo "${PID}" > "${ROOT}/run/llama-server-${PORT}.pid"
trap 'kill "${PID}" 2>/dev/null || true' EXIT INT TERM
wait "${PID}"
