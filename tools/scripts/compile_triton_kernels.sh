#!/bin/bash
set -e

REPO_ROOT=$(pwd)
OUT_DIR="${REPO_ROOT}/build/precompiled_triton_kernels"

# Create the output directory if it doesn't exist
mkdir -p "${OUT_DIR}"

# List of Triton kernel scripts
TRITON_KERNEL_SCRIPTS=(
  "onnxruntime/contrib_ops/cuda/my_triton_kernel.py"
)

python3 -m venv "${REPO_ROOT}/triton_env"
source "${REPO_ROOT}/triton_env/bin/activate"
pip install -r "${REPO_ROOT}/tools/ci_build/compile_triton_requirements.txt"

# Compile each Triton kernel script
python3 "${REPO_ROOT}/tools/ci_build/compile_triton.py" \
    --header "${OUT_DIR}/triton_kernel_infos.h" \
    --script_files "${TRITON_KERNEL_SCRIPTS[@]}" \
    --obj_file "${OUT_DIR}/triton_kernel_infos.a"
