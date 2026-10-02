#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo "Usage: bash build_runtime.sh GENAI_SOURCE ORT_HOME CUDA_TOOLKIT CUDNN_LIB DEPS_DIR CUDA_ARCH NEW_BUILD_DIR" >&2
  echo "Requires prepare_sources.sh output, matching external fixtures and explicit PYTHON interpreter." >&2
  exit 2
fi
source=$1 ort=$2 toolkit=$3 cudnn=$4 deps=$5 arch=$6 build=$7
here=$(cd "$(dirname "$0")" && pwd)
: "${PYTHON:?Set PYTHON to an existing Python 3.11+ interpreter}"
for path in "$source" "$ort" "$toolkit" "$cudnn" "$deps" "$build"; do
  [[ "$path" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ ! -e "$build" ]] || { echo "Refusing existing build directory" >&2; exit 2; }
[[ "$arch" =~ ^[0-9]+$ ]] || { echo "Explicit numeric CUDA architecture required (historical A100: 80)" >&2; exit 2; }
test "$(git -C "$source" rev-parse HEAD)" = ed5f4e87147731e5b07810f9f5c90103b3603cdf
grep -q logits_allocation_tests "$source/test/CMakeLists.txt" || {
  echo "BLOCKED: apply included historical-tests-normalized.patch using prepare_sources.sh first" >&2
  exit 2
}
printf '%s  %s\n' \
  30ac48501a07b7d3a3a71b5dbd51c52928ac9177a25cb1c38c371a8b4d164ed8 \
  "$source/test/logits_allocation/tests.cpp" \
  82ae59b4cf7751f5a71ae41f4136fbaebd5e39a40f34dcd2f2a5b1c99dec0731 \
  "$source/test/CMakeLists.txt" | sha256sum --check --status
git -C "$source" diff --cached --quiet
production_diff=$(git -C "$source" diff -- src)
if [[ -n "$production_diff" && "$production_diff" != "$(cat "$here/production.diff")" ]]; then
  echo "BLOCKED: unexpected production source changes" >&2
  exit 2
fi
"$PYTHON" -B "$here/allocation.py" check-fixtures --models-dir "$source/test/models"
test -f "$ort/include/onnxruntime_experimental_c_api.inc"
grep -q 'OrtApi_DebugLogAndShrinkGpuArenas' "$ort/include/onnxruntime_experimental_c_api.inc"
args=()
for name in googletest onnxruntime_extensions gsl nlohmann_json dr_libs dlib; do
  test -d "$deps/$name-src"
  args+=("-DFETCHCONTENT_SOURCE_DIR_${name^^}=$deps/$name-src")
done
export CUDACXX="$toolkit/bin/nvcc"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$build:$ort/lib:$toolkit/lib64:$cudnn"
cmake -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DORT_HOME="$ort" \
  -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER="$toolkit/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_CUDA_FLAGS=--threads=1 \
  -DENABLE_PYTHON=OFF -DENABLE_TESTS=ON -DENABLE_CUDA_KERNEL_TESTS=OFF \
  -DENABLE_MODEL_BENCHMARK=OFF -DENABLE_TELEMETRY=OFF -DENABLE_TRACING=OFF -DUSE_GUIDANCE=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON "${args[@]}"
cmake --build "$build" --target onnxruntime-genai onnxruntime-genai-cuda \
  logits_allocation_tests unit_tests --parallel 2
sha256sum "$build/libonnxruntime-genai.so" "$build/libonnxruntime-genai-cuda.so" \
  "$build/logits_allocation_tests" "$build/unit_tests" > "$build/runtime-hashes.sha256"
