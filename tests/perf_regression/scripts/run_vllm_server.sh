#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_vllm_server.sh [text_10c|vl_small|vl_large]
# Default: text_10c
SCENARIO=${1:-text_10c}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${PERF_REGRESSION_ROOT:-$(dirname "${SCRIPT_DIR}")}

case "${SCENARIO}" in
  text_10c)
    DEFAULT_MODEL=/home/common/model/qwen3/Qwen3-1.7B
    DEFAULT_PORT=18110
    DEFAULT_MAX_LEN=12288
    DEFAULT_MAX_SEQS=16
    ;;
  vl_small|vl_large)
    DEFAULT_MODEL=/home/common/model/qwen3/Qwen3-VL-4B-Instruct
    DEFAULT_PORT=18120
    DEFAULT_MAX_LEN=8192
    DEFAULT_MAX_SEQS=4
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    echo "Usage: bash run_vllm_server.sh [text_10c|vl_small|vl_large]" >&2
    exit 1
    ;;
esac

MODEL=${VLLM_MODEL:-${DEFAULT_MODEL}}
PORT=${PERF_SERVER_PORT:-${DEFAULT_PORT}}
MAX_LEN=${PERF_VLLM_MAX_LEN:-${DEFAULT_MAX_LEN}}
MAX_SEQS=${PERF_VLLM_MAX_SEQS:-${DEFAULT_MAX_SEQS}}

export ASCEND_RT_VISIBLE_DEVICES=${PERF_NPU_ID:-7}

mkdir -p "${ROOT}/logs" "${ROOT}/run"
LOG=${ROOT}/logs/vllm-server-${PORT}.log

VLLM_ARGS=(vllm serve "${MODEL}")
if [[ "${SCENARIO}" == vl_small || "${SCENARIO}" == vl_large ]]; then
  VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 1}')
fi

"${VLLM_ARGS[@]}" \
  --dtype float16 \
  --port "${PORT}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --enable-prefix-caching \
  --gpu-memory-utilization "${PERF_VLLM_GPU_MEM_UTIL:-0.85}" \
  2>&1 | tee "${LOG}" &
PID=$!
echo "${PID}" > "${ROOT}/run/vllm-server-${PORT}.pid"
trap 'kill "${PID}" 2>/dev/null || true' EXIT INT TERM
wait "${PID}"
