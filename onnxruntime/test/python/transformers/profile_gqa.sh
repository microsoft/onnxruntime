#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Profile the CUDA GroupQueryAttention decode path with nsys.
#
# Usage:
#   ./profile_gqa.sh --all
#   ./profile_gqa.sh --fp16 --int8
#   ./profile_gqa.sh --fp16 --past-sequence-length 8192 --local-window-size 128
#   ./profile_gqa.sh --bf16 --num-heads 64 --kv-num-heads 8 --head-size 128
#   ./profile_gqa.sh --fp16 --compare-qk-norm --past-sequence-length 2048
#   NSYS=~/cuda13.0/bin/nsys ./profile_gqa.sh --fp16 --qk-norm
#   CUDA_VISIBLE_DEVICES=1 PYTHON=python3 ./profile_gqa.sh --int4
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
NSYS="${NSYS:-}"

# Parse arguments
RUN_FP16=false
RUN_INT8=false
RUN_INT4=false
RUN_INT8_QUANT=false
RUN_BF16=false

# Profile parameters to pass through to profile_gqa.py
BATCH_SIZE=""
SEQUENCE_LENGTH=""
PAST_SEQUENCE_LENGTH=""
MAX_SEQUENCE_LENGTH=""
PACKED_QKV=""
SHARE_KV_SCALE=""
NUM_HEADS=""
KV_NUM_HEADS=""
HEAD_SIZE=""
LOCAL_WINDOW_SIZE=""
HEAD_SINK=""
NO_ROTARY=""
QK_NORM=false
COMPARE_QK_NORM=false
QK_NORM_EPSILON=""
WARMUP=5
REPEAT=100
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fp16)
            RUN_FP16=true
            echo "==== 🚀 FP16 run enabled ===="
            ;;
        --int8)
            RUN_INT8=true
            echo "==== 🚀 INT8 run enabled ===="
            ;;
        --int4)
            RUN_INT4=true
            echo "==== 🚀 INT4 run enabled ===="
            ;;
        --int8_quant)
            RUN_INT8_QUANT=true
            echo "==== 🚀 INT8 Quant run enabled ===="
            ;;
        --bf16)
            RUN_BF16=true
            echo "==== 🚀 BF16 run enabled ===="
            ;;
        --all)
            RUN_FP16=true
            RUN_INT8=true
            RUN_INT4=true
            RUN_INT8_QUANT=true
            RUN_BF16=true
            echo "==== 🚀 All runs enabled ===="
            ;;
        --qk-norm)
            QK_NORM=true
            echo "==== QK-Norm enabled ===="
            ;;
        --compare-qk-norm)
            COMPARE_QK_NORM=true
            echo "==== Compare baseline vs QK-Norm enabled ===="
            ;;
        --qk-norm-epsilon)
            QK_NORM_EPSILON="--qk-norm-epsilon $2"
            echo "==== QK-Norm epsilon: $2 ===="
            shift
            ;;
        --warmup)
            WARMUP="$2"
            echo "==== Warmup iterations: $2 ===="
            shift
            ;;
        --repeat)
            REPEAT="$2"
            echo "==== Repeat iterations: $2 ===="
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="--batch-size $2"
            echo "==== Batch size: $2 ===="
            shift
            ;;
        -s|--sequence-length)
            SEQUENCE_LENGTH="--sequence-length $2"
            echo "==== Sequence length: $2 ===="
            shift
            ;;
        -p|--past-sequence-length)
            PAST_SEQUENCE_LENGTH="--past-sequence-length $2"
            echo "==== Past sequence length: $2 ===="
            shift
            ;;
        --max-sequence-length)
            MAX_SEQUENCE_LENGTH="--max-sequence-length $2"
            echo "==== Max sequence length: $2 ===="
            shift
            ;;
        --qkv)
            PACKED_QKV="--is-packed-qkv"
            echo "==== Packed QKV enabled ===="
            ;;
        --share-kv-scale)
            SHARE_KV_SCALE="--share-kv-scale"
            echo "==== Share KV scale enabled ===="
            ;;
        --num-heads)
            NUM_HEADS="--num-heads $2"
            echo "==== Num Heads: $2 ===="
            shift
            ;;
        --kv-num-heads)
            KV_NUM_HEADS="--kv-num-heads $2"
            echo "==== KV Num Heads: $2 ===="
            shift
            ;;
        --head-size)
            HEAD_SIZE="--head-size $2"
            echo "==== Head size: $2 ===="
            shift
            ;;
        -w|--local-window-size)
            LOCAL_WINDOW_SIZE="--local-window-size $2"
            echo "==== Local window size: $2 ===="
            shift
            ;;
        --head-sink)
            HEAD_SINK="--head-sink"
            echo "==== Head sink enabled ===="
            ;;
        --no-rotary)
            NO_ROTARY="--no-rotary"
            echo "==== Rotary disabled ===="
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Build extra args string
EXTRA_ARGS="${BATCH_SIZE} ${SEQUENCE_LENGTH} ${PAST_SEQUENCE_LENGTH} ${MAX_SEQUENCE_LENGTH} ${PACKED_QKV} ${SHARE_KV_SCALE} ${NUM_HEADS} ${KV_NUM_HEADS} ${HEAD_SIZE} ${LOCAL_WINDOW_SIZE} ${HEAD_SINK} ${NO_ROTARY} ${QK_NORM_EPSILON}"

if [[ -z "${NSYS}" ]]; then
    if command -v nsys >/dev/null; then
        NSYS="$(command -v nsys)"
    elif [[ -x "${HOME}/cuda13.0/bin/nsys" ]]; then
        NSYS="${HOME}/cuda13.0/bin/nsys"
    fi
fi

if [[ -z "${NSYS}" || ! -x "${NSYS}" ]]; then
    echo "Error: nsys not found. Install NVIDIA Nsight Systems or add it to PATH."
    echo "       Or set NSYS=/path/to/nsys (for example NSYS=~/cuda13.0/bin/nsys)."
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

# profile_one <mode> <tag> <output_base> <extra_cli_args> [env_var=value ...]
profile_one() {
    local mode="$1"
    local tag="$2"
    local base="$3"
    local cli_extra="$4"
    shift 4

    local env_args=()
    local e
    for e in "$@"; do
        env_args+=(-e "${e}")
    done

    echo ""
    echo "---- Profiling ${mode} ----"
    rm -f "${base}.nsys-rep" "${base}.sqlite"
    "${NSYS}" profile -t cuda,nvtx --force-overwrite true "${env_args[@]}" -o "${base}" --export=sqlite \
        "${PY}" "${SCRIPT_DIR}/profile_gqa.py" --mode "${mode}" --warmup "${WARMUP}" --repeat "${REPEAT}" ${EXTRA_ARGS} ${cli_extra}

    echo ""
    echo "---- Kernel results (${mode}) ----"
    local range_name="benchmark_${mode}"
    if [[ "${cli_extra}" == *"--qk-norm"* ]]; then
        range_name="benchmark_${mode}_qknorm"
    fi
    if [[ "${HAVE_NVTX}" -eq 1 ]]; then
        "${PY}" "${SCRIPT_DIR}/parse_nsys.py" "${base}.sqlite" --nvtx-range "${range_name}" --tag "${tag}" \
            --pattern "%onnxruntime%" --pattern "%cudnn%"
    else
        "${PY}" "${SCRIPT_DIR}/parse_nsys.py" "${base}.sqlite" --skip-first "${WARMUP}" --tag "${tag}" \
            --pattern "%onnxruntime%" --pattern "%cudnn%"
    fi
}

profile_mode() {
    local mode="$1"
    local tag="$2"
    local base="$3"
    shift 3

    if [[ "${COMPARE_QK_NORM}" == true ]]; then
        profile_one "${mode}" "${tag}" "${base}" "" "$@"
        profile_one "${mode}" "${tag}QK" "${base}_qknorm" "--qk-norm" "$@"
    elif [[ "${QK_NORM}" == true ]]; then
        profile_one "${mode}" "${tag}QK" "${base}_qknorm" "--qk-norm" "$@"
    else
        profile_one "${mode}" "${tag}" "${base}" "" "$@"
    fi
}

if [ "$RUN_FP16" = true ]; then
    profile_mode fp16 Fp16 gqa_fp16
fi

if [ "$RUN_BF16" = true ]; then
    profile_mode bf16 Bf16 gqa_bf16
fi

if [ "$RUN_INT8" = true ]; then
    profile_mode int8 Int8 gqa_int8 ORT_FLASH_ATTENTION_QUERY_DYNAMIC_QUANT=0
fi

if [ "$RUN_INT8_QUANT" = true ]; then
    profile_mode int8 Int8Q gqa_int8_quant ORT_FLASH_ATTENTION_QUERY_DYNAMIC_QUANT=1
fi

if [ "$RUN_INT4" = true ]; then
    profile_mode int4 Int4 gqa_int4
fi
