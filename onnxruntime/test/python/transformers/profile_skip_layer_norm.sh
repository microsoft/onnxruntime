#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Profile SkipLayerNormalization CUDA kernel with nsys.
#
# Usage:
#   ./profile_skip_layer_norm.sh                           # Profile with defaults (fp16, H=4096)
#   ./profile_skip_layer_norm.sh --hidden-size 8192        # Different hidden size
#   ./profile_skip_layer_norm.sh --fp32 --batch-size 4     # FP32 mode, batch=4
#

set -e
set -o pipefail

# Default parameters
BATCH_SIZE=""
SEQ_LEN=""
HIDDEN_SIZE=""
MODE="--mode fp16"
SIMPLIFIED=""
OUTPUT_NAME="sln_profile"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            shift
            ;;
        --seq-len)
            SEQ_LEN="--seq-len $2"
            shift
            ;;
        --hidden-size)
            HIDDEN_SIZE="--hidden-size $2"
            shift
            ;;
        --fp32)
            MODE="--mode fp32"
            ;;
        --simplified)
            SIMPLIFIED="--simplified"
            ;;
        -o|--output)
            OUTPUT_NAME="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch-size N] [--seq-len N] [--hidden-size N] [--fp32] [--simplified] [-o NAME]"
            exit 1
            ;;
    esac
    shift
done

EXTRA_ARGS="${BATCH_SIZE} ${SEQ_LEN} ${HIDDEN_SIZE} ${MODE} ${SIMPLIFIED}"

pip install nvtx 2>/dev/null || true

echo ""
echo "========================================"
echo "  Profiling: SkipLayerNormalization"
echo "========================================"
rm -f "${OUTPUT_NAME}.nsys-rep" "${OUTPUT_NAME}.sqlite"
nsys profile -o "${OUTPUT_NAME}" --export=sqlite \
    python profile_skip_layer_norm.py --warmup 5 --repeat 100 $EXTRA_ARGS
echo ""
echo "---- Kernel results ----"
python parse_nsys.py "${OUTPUT_NAME}.sqlite" --skip-first 5

echo ""
echo "Done."
