#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_llama_server.sh [text_10c|vl_small|vl_large]
# Default: text_10c
SCENARIO=${1:-text_10c}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${PERF_REGRESSION_ROOT:-$(dirname "${SCRIPT_DIR}")}
REPO_ROOT=${LLAMA_SOURCE:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
BUILD_DIR=${LLAMA_BUILD_DIR:-${REPO_ROOT}/build-cann-aclgraph}

# Source CANN and ATB environments so the server can find libascendcl and ATB.
ASCEND_HOME=${ASCEND_HOME:-/usr/local/Ascend}
CANN_SET_ENV=${CANN_SET_ENV:-}
ATB_SET_ENV=${ATB_SET_ENV:-}

if [[ -z "${CANN_SET_ENV}" && -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  CANN_SET_ENV="${ASCEND_TOOLKIT_HOME}/set_env.sh"
fi
if [[ -z "${CANN_SET_ENV}" && -f "${ASCEND_HOME}/ascend-toolkit/set_env.sh" ]]; then
  CANN_SET_ENV="${ASCEND_HOME}/ascend-toolkit/set_env.sh"
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

# Per-scenario defaults (overridable by env vars)
case "${SCENARIO}" in
  text_10c)
    DEFAULT_MODEL=${LLAMA_TEXT_MODEL:-${LLAMA_MODEL:-}}
    DEFAULT_MMPROJ=""
    DEFAULT_PORT=18112
    DEFAULT_CTX=8192
    DEFAULT_PARALLEL=10
    DEFAULT_ACT_TYPE=f16
    ;;
  vl_small|vl_large)
    DEFAULT_MODEL=${LLAMA_VL_MODEL:-${LLAMA_MODEL:-}}
    DEFAULT_MMPROJ=${LLAMA_VL_MMPROJ:-${LLAMA_MMPROJ:-}}
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
if [[ -z "${MODEL}" ]]; then
  echo "llama model path is not set; use LLAMA_MODEL, LLAMA_TEXT_MODEL or LLAMA_VL_MODEL" >&2
  exit 1
fi
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
MMPROJ_OFFLOAD_ARGS=()
if [[ "${PERF_LLAMA_MMPROJ_OFFLOAD:-0}" != "1" ]]; then
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
  -ngl 99 \
  -sm none \
  -mg 0 \
  --kv-paged \
  --cache-ram 0 \
  -kvbls 128 \
  -b 2048 \
  -ub 2048 \
  "${ACT_ARGS[@]}" \
  "${MMPROJ_OFFLOAD_ARGS[@]}" \
  --metrics \
  "${MMPROJ_ARGS[@]}" \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${ROOT}/run/llama-server-${PORT}.pid"
trap 'kill "${PID}" 2>/dev/null || true' EXIT INT TERM
wait "${PID}"
