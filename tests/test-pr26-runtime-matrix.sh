#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-build/bin/test-backend-ops}"
OUT_DIR="${2:-build/pr26-runtime-matrix}"
mkdir -p "${OUT_DIR}"

run_case() {
    local dense="$1"
    local moe="$2"
    local op="$3"
    local log="${OUT_DIR}/dense${dense}-moe${moe}-${op}.log"
    env -u GGML_CANN_MOE_FULL_FUSION \
        GGML_CANN_W8A8_A8="${dense}" \
        GGML_CANN_MOE_W8A8_A8="${moe}" \
        GGML_CANN_MOE_DEBUG=1 \
        GGML_CANN_MOE_FULL_DEBUG=1 \
        "${BIN}" -b CANN0 -o "${op}" -p type_a=w8a8 >"${log}" 2>&1
    if [ "${op}" = "MUL_MAT" ]; then
        grep -q '6/6 tests passed' "${log}"
    else
        grep -q '5/5 tests passed' "${log}"
    fi
    printf 'dense=%s moe=%s %-10s PASS\n' "${dense}" "${moe}" "${op}"
}

for dense in 0 1; do
    for moe in 0 1; do
        run_case "${dense}" "${moe}" MUL_MAT
        run_case "${dense}" "${moe}" MUL_MAT_ID
        log="${OUT_DIR}/dense${dense}-moe${moe}-MUL_MAT_ID.log"
        if [ "${dense}" = "1" ] && [ "${moe}" = "1" ]; then
            grep -q 'supported=1' "${log}"
            if grep -q '\[MoE FULL' "${log}"; then
                echo "unexpected full-fusion attempt" >&2
                exit 1
            fi
        else
            if grep -q 'supported=1' "${log}"; then
                echo "unexpected grouped route activation" >&2
                exit 1
            fi
        fi
    done
done

echo "PASS: dense/MoE flag matrix and full-fusion bypass"
