#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMA_SOURCE=${LLAMA_SOURCE:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
BUILD_DIR=${LLAMA_BUILD_DIR:-${LLAMA_SOURCE}/build-cann-aclgraph}

cmake -S "${LLAMA_SOURCE}" -B "${BUILD_DIR}" \
  -DGGML_CANN=ON \
  -DCANN_SOC_VERSION=310P3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ACL_GRAPH=ON \
  -DLLAMA_CURL=OFF
cmake --build "${BUILD_DIR}" -j"${PERF_BUILD_JOBS:-8}" --target llama-server

echo "llama-server: ${BUILD_DIR}/bin/llama-server"
