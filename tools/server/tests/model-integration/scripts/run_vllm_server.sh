#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_vllm_server.sh [text_10c|vl_small|vl_large] [--config PATH] [--model PATH]
# Default: text_10c
SCENARIO=text_10c
CONFIG_FILE=""
CLI_MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a path" >&2; exit 1; }
      CLI_MODEL="$2"
      shift 2
      ;;
    *)
      SCENARIO="$1"
      shift
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${PERF_REGRESSION_ROOT:-$(dirname "${SCRIPT_DIR}")}
CONFIG_FILE=${CONFIG_FILE:-${SCRIPT_DIR}/../config.json}
eval "$(python3 "${SCRIPT_DIR}/test_config.py" --config "${CONFIG_FILE}" --engine vllm --scenario "${SCENARIO}" --shell)"

case "${SCENARIO}" in
  text_10c)
    DEFAULT_MODEL="${CFG_VLLM_TEXT_MODEL}"
    ;;
  vl_small|vl_large)
    DEFAULT_MODEL="${CFG_VLLM_VL_MODEL}"
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    echo "Usage: bash run_vllm_server.sh [text_10c|vl_small|vl_large]" >&2
    exit 1
    ;;
esac

MODEL=${CLI_MODEL:-${DEFAULT_MODEL}}
PORT=${PERF_SERVER_PORT:-${CFG_PORT}}
MAX_LEN=${PERF_VLLM_MAX_LEN:-${CFG_MAX_LEN}}
MAX_SEQS=${PERF_VLLM_MAX_SEQS:-${CFG_MAX_SEQS}}

export ASCEND_RT_VISIBLE_DEVICES=${PERF_NPU_ID:-${CFG_NPU_ID}}

mkdir -p "${ROOT}/logs" "${ROOT}/run"
LOG=${ROOT}/logs/vllm-server-${PORT}.log

VLLM_ARGS=(vllm serve "${MODEL}")
if [[ -n "${CFG_LIMIT_MM}" && "${CFG_LIMIT_MM}" != "{}" ]]; then
  VLLM_ARGS+=(--limit-mm-per-prompt "${CFG_LIMIT_MM}")
fi
if [[ "${CFG_PREFIX_CACHING}" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

"${VLLM_ARGS[@]}" \
  --dtype "${CFG_DTYPE}" \
  --port "${PORT}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --gpu-memory-utilization "${PERF_VLLM_GPU_MEM_UTIL:-${CFG_GPU_MEM_UTIL}}" \
  --compilation-config "${CFG_COMPILATION_CONFIG}" \
  2>&1 | tee "${LOG}" &
PID=$!
echo "${PID}" > "${ROOT}/run/vllm-server-${PORT}.pid"
trap 'kill "${PID}" 2>/dev/null || true' EXIT INT TERM
wait "${PID}"
