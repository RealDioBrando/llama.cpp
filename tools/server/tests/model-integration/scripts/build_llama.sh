#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${SCRIPT_DIR}/../config.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash build_llama.sh [--config PATH]" >&2
      exit 1
      ;;
  esac
done
eval "$(python3 "${SCRIPT_DIR}/test_config.py" --config "${CONFIG_FILE}" --engine build --shell)"
LLAMA_SOURCE=${LLAMA_SOURCE:-${CFG_LLAMA_SOURCE:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}
BUILD_DIR=${LLAMA_BUILD_DIR:-${CFG_LLAMA_BUILD_DIR:-${LLAMA_SOURCE}/build-cann-aclgraph}}

cmake -S "${LLAMA_SOURCE}" -B "${BUILD_DIR}" \
  -DGGML_CANN="${CFG_BUILD_GGML_CANN}" \
  -DCANN_SOC_VERSION="${CFG_BUILD_SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE="${CFG_BUILD_TYPE}" \
  -DUSE_ACL_GRAPH="${CFG_BUILD_USE_ACL_GRAPH}" \
  -DLLAMA_CURL="${CFG_BUILD_LLAMA_CURL}"
cmake --build "${BUILD_DIR}" -j"${PERF_BUILD_JOBS:-${CFG_BUILD_JOBS}}" --target llama-server

echo "llama-server: ${BUILD_DIR}/bin/llama-server"
