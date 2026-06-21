#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Profile the CUDA QMoE GEMV decode path with nsys.
#
# Usage:
#   ./profile_qmoe_gemv.sh
#   ./profile_qmoe_gemv.sh --list-cases
#   ./profile_qmoe_gemv.sh --case m8_top2_fp16_128x256 --warmup 5 --repeat 200
#   ./profile_qmoe_gemv.sh --case gpt_oss_20b_m1_top4_fp16_2880x2880_e32 --warmup 5 --repeat 100
#   ./profile_qmoe_gemv.sh --batch-size 1 --sequence-length 1 --hidden-size 1024 --intermediate-size 4096 --num-experts 8 --top-k 2 --quant-bits 8 --block-size 128
#   CUDA_VISIBLE_DEVICES=1 ./profile_qmoe_gemv.sh -o /tmp/qmoe_gemv
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="m1_top2_fp16_128x256"
WARMUP=5
REPEAT=100
OUTPUT_NAME="qmoe_gemv_profile"
PY="${PYTHON:-python}"
EXTRA_ARGS=()
LIST_CASES=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --case)
            CASE="$2"
            shift
            ;;
        --list-cases)
            LIST_CASES=1
            ;;
        --batch-size|--sequence-length|--hidden-size|--intermediate-size|--num-experts|--top-k|--dtype|--quant-bits|--block-size)
            EXTRA_ARGS+=("$1" "$2")
            shift
            ;;
        --repeat)
            REPEAT="$2"
            shift
            ;;
        --warmup)
            WARMUP="$2"
            shift
            ;;
        --python)
            PY="$2"
            shift
            ;;
        -o|--output)
            OUTPUT_NAME="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--list-cases] [--case NAME] [--batch-size N] [--sequence-length N] [--hidden-size N] [--intermediate-size N] [--num-experts N] [--top-k N] [--dtype FLOAT16|BFLOAT16] [--quant-bits 4|8] [--block-size 0|32|64|128] [--warmup N] [--repeat N] [--python PYTHON] [-o NAME]"
            exit 1
            ;;
    esac
    shift
done

if [[ "${LIST_CASES}" -eq 1 ]]; then
    "${PY}" "${SCRIPT_DIR}/profile_qmoe_gemv.py" --list-cases
    exit 0
fi

if ! command -v nsys >/dev/null; then
    echo "Error: nsys not found. Install NVIDIA Nsight Systems or add it to PATH."
    exit 1
fi

HAVE_NVTX=0
if "${PY}" -c "import nvtx" 2>/dev/null; then
    HAVE_NVTX=1
else
    echo "Note: 'nvtx' package not installed. NVTX range markers will be disabled."
    echo "      Install with: pip install nvtx"
    echo "      Falling back to --skip-first to exclude warmup-like first calls."
fi

echo ""
echo "========================================"
echo "  Profiling: CUDA QMoE GEMV"
echo "========================================"
echo "Case:   ${CASE}"
echo "Warmup: ${WARMUP}"
echo "Repeat: ${REPEAT}"
echo "Python: ${PY}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "Custom args: ${EXTRA_ARGS[*]}"
fi

profile_one() {
    local mode="$1"
    local disable_arg=""
    local base="${OUTPUT_NAME}_${mode}"
    if [[ "${mode}" == "gemm" ]]; then
        disable_arg="--disable-gemv"
    fi

    echo ""
    echo "---- Profiling ${mode} ----"
    rm -f "${base}.nsys-rep" "${base}.sqlite"
    nsys profile -t cuda,nvtx --force-overwrite true -o "${base}" --export=sqlite \
        "${PY}" "${SCRIPT_DIR}/profile_qmoe_gemv.py" \
            --case "${CASE}" "${EXTRA_ARGS[@]}" --warmup "${WARMUP}" --repeat "${REPEAT}" --nvtx ${disable_arg}

    echo ""
    echo "---- Kernel results (${mode}) ----"
    if [[ "${HAVE_NVTX}" -eq 1 ]]; then
        "${PY}" "${SCRIPT_DIR}/parse_nsys.py" "${base}.sqlite" --nvtx-range benchmark
    else
        "${PY}" "${SCRIPT_DIR}/parse_nsys.py" "${base}.sqlite" --skip-first 1
    fi
}

profile_one gemv
profile_one gemm

echo ""
echo "Done."
