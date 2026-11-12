#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_llama_server.sh [text_10c|vl_small|vl_large] [--config PATH] [--model PATH] [--mmproj PATH]
# Default: text_10c
SCENARIO=text_10c
CONFIG_FILE=""
CLI_MODEL=""
CLI_MMPROJ=""
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
    --mmproj)
      [[ $# -ge 2 ]] || { echo "--mmproj requires a path" >&2; exit 1; }
      CLI_MMPROJ="$2"
      shift 2
      ;;
    *)
      SCENARIO="$1"
      shift
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-${SCRIPT_DIR}/../config.json}
ROOT=${PERF_REGRESSION_ROOT:-$(dirname "${SCRIPT_DIR}")}
REPO_ROOT=${LLAMA_SOURCE:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}
BUILD_DIR=${LLAMA_BUILD_DIR:-${REPO_ROOT}/build-cann-aclgraph}

eval "$(python3 "${SCRIPT_DIR}/test_config.py" --config "${CONFIG_FILE}" --engine llama --scenario "${SCENARIO}" --shell)"
REPO_ROOT=${LLAMA_SOURCE:-${CFG_LLAMA_SOURCE:-${REPO_ROOT}}}
BUILD_DIR=${LLAMA_BUILD_DIR:-${CFG_LLAMA_BUILD_DIR:-${REPO_ROOT}/build-cann-aclgraph}}

# Source CANN and ATB environments so the server can find libascendcl and ATB.
ASCEND_HOME=${ASCEND_HOME:-${CFG_ASCEND_HOME}}
CANN_SET_ENV=${CANN_SET_ENV:-${CFG_CANN_SET_ENV}}
ATB_SET_ENV=${ATB_SET_ENV:-${CFG_ATB_SET_ENV}}

if [[ -z "${CANN_SET_ENV}" && -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  CANN_SET_ENV="${ASCEND_TOOLKIT_HOME}/set_env.sh"
fi
if [[ -z "${CANN_SET_ENV}" && -f "${ASCEND_HOME}/ascend-toolkit/set_env.sh" ]]; then
  CANN_SET_ENV="${ASCEND_HOME}/ascend-toolkit/set_env.sh"
fi
if [[ -z "${CANN_SET_ENV}" && -f "${ASCEND_HOME}/cann-9.1.0/set_env.sh" ]]; then
  CANN_SET_ENV="${ASCEND_HOME}/cann-9.1.0/set_env.sh"
fi
if [[ -z "${ATB_SET_ENV}" && -f "${ASCEND_HOME}/nnal/atb/set_env.sh" ]]; then
  ATB_SET_ENV="${ASCEND_HOME}/nnal/atb/set_env.sh"
fi

if [[ -n "${CANN_SET_ENV}" && -f "${CANN_SET_ENV}" ]]; then
  set +u
  source "${CANN_SET_ENV}"
  set -u
fi

if [[ -n "${ATB_SET_ENV}" && -f "${ATB_SET_ENV}" ]]; then
  set +u
  source "${ATB_SET_ENV}"
  set -u
fi

# Keep these as fallbacks for images where the env scripts are not installed.
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${ATB_HOME_PATH:-}" && -d "${ATB_HOME_PATH}/lib" ]]; then
  export LD_LIBRARY_PATH="${ATB_HOME_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi

case "${SCENARIO}" in
  text_10c)
    DEFAULT_MODEL="${CFG_LLAMA_TEXT_MODEL}"
    DEFAULT_MMPROJ=""
    ;;
  vl_small|vl_large)
    DEFAULT_MODEL="${CFG_LLAMA_VL_MODEL}"
    DEFAULT_MMPROJ="${CFG_LLAMA_VL_MMPROJ}"
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    echo "Usage: bash run_llama_server.sh [text_10c|vl_small|vl_large]" >&2
    exit 1
    ;;
esac

MODEL=${CLI_MODEL:-${DEFAULT_MODEL}}
PORT=${PERF_SERVER_PORT:-${CFG_PORT}}
CTX=${PERF_LLAMA_CTX:-${CFG_CTX}}
PARALLEL=${PERF_LLAMA_PARALLEL:-${CFG_PARALLEL}}
THREADS=${PERF_LLAMA_THREADS:-${CFG_THREADS}}
THREADS_BATCH=${PERF_LLAMA_THREADS_BATCH:-${CFG_THREADS_BATCH}}
MMPROJ=${CLI_MMPROJ:-${DEFAULT_MMPROJ}}
MMPROJ_ARGS=(--mmproj-auto)
if [[ -n "${MMPROJ}" ]]; then
  MMPROJ_ARGS=(--mmproj "${MMPROJ}")
fi

export ASCEND_RT_VISIBLE_DEVICES=${PERF_NPU_ID:-${CFG_NPU_ID}}
export GGML_CANN_ACL_GRAPH=${PERF_LLAMA_ACL_GRAPH:-${CFG_ACL_GRAPH}}
export GGML_CANN_PREFILL_USE_GRAPH=${PERF_LLAMA_PREFILL_GRAPH:-${CFG_PREFILL_GRAPH}}
export GGML_CANN_FA_V5=${CFG_FA_V5}
export GGML_CANN_WEIGHT_NZ=${CFG_WEIGHT_NZ}
export GGML_CANN_GDN_PROJ=${CFG_GDN_PROJ}
export GGML_CANN_GDN_SMALL=${CFG_GDN_SMALL}
export GGML_CANN_GDN_PREFILL_CHUNK=${CFG_GDN_PREFILL_CHUNK}

mkdir -p "${ROOT}/logs" "${ROOT}/run"
LOG=${ROOT}/logs/llama-server-${PORT}.log

ACT_TYPE=${PERF_LLAMA_ACT_TYPE:-${CFG_ACT_TYPE}}
ACT_ARGS=()
if [[ "${ACT_TYPE}" != "none" ]]; then
  ACT_ARGS=(-at "${ACT_TYPE}")
fi
KV_PAGED_ARGS=()
if [[ "${CFG_KV_PAGED}" == "1" ]]; then
  KV_PAGED_ARGS=(--kv-paged)
fi
CACHE_RAM_ARGS=(--cache-ram "${CFG_CACHE_RAM}")
MMPROJ_OFFLOAD_ARGS=()
if [[ "${PERF_LLAMA_MMPROJ_OFFLOAD:-${CFG_MMPROJ_OFFLOAD}}" != "1" ]]; then
  MMPROJ_OFFLOAD_ARGS=(--no-mmproj-offload)
fi

"${BUILD_DIR}/bin/llama-server" \
  -m "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -c "${CTX}" \
  -np "${PARALLEL}" \
  -t "${THREADS}" \
  -tb "${THREADS_BATCH}" \
  -ngl "${CFG_NGPU_LAYERS}" \
  -sm "${CFG_SAMPLING_MODE}" \
  -mg "${CFG_MAIN_GPU}" \
  "${KV_PAGED_ARGS[@]}" \
  "${CACHE_RAM_ARGS[@]}" \
  -kvbls "${CFG_KV_BLOCK_SIZE}" \
  -b "${CFG_BATCH_SIZE}" \
  -ub "${CFG_UBATCH_SIZE}" \
  "${ACT_ARGS[@]}" \
  "${MMPROJ_OFFLOAD_ARGS[@]}" \
  --metrics \
  "${MMPROJ_ARGS[@]}" \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${ROOT}/run/llama-server-${PORT}.pid"
cleanup() {
  kill "${PID}" 2>/dev/null || true
  wait "${PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait "${PID}"
