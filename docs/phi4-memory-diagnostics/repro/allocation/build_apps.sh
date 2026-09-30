#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "Usage: bash build_apps.sh ORT_HOME GENAI_SOURCE GENAI_LIBRARY_DIR CUDA_TOOLKIT NEW_BUILD_DIR" >&2
  exit 2
fi
ort=$1 source=$2 libraries=$3 toolkit=$4 build=$5
for path in "$ort" "$source" "$libraries" "$toolkit" "$build"; do
  [[ "$path" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ ! -e "$build" ]] || { echo "Refusing existing build directory" >&2; exit 2; }
test -f "$source/src/ort_genai.h"
test -f "$ort/include/onnxruntime_experimental_c_api.inc"
grep -q 'OrtApi_DebugLogAndShrinkGpuArenas' "$ort/include/onnxruntime_experimental_c_api.inc"
test -f "$libraries/libonnxruntime-genai.so"
test -x "$toolkit/bin/nvcc"
cmake -S "$(cd "$(dirname "$0")" && pwd)/native" -B "$build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DORT_HOME="$ort" \
  -DGENAI_SOURCE_DIR="$source" -DGENAI_LIBRARY_DIR="$libraries" -DCUDAToolkit_ROOT="$toolkit"
cmake --build "$build" --target phi_phase_a phi_phase_b --parallel 2
sha256sum "$build/phi_phase_a" "$build/phi_phase_b" > "$build/app-hashes.sha256"
readelf -d "$build/phi_phase_b" > "$build/phase-b-dynamic.txt"
