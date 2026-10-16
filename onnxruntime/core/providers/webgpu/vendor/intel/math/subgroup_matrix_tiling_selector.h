// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Workgroup-memory (SLM) budget, in f16 elements, that the selector below fits
// tilings into. 16384 f16 == 32 KB, the common WebGPU maxComputeWorkgroupStorageSize.
// Shared with (not duplicated by) the Conv kernel's own split-K safety clamp, since
// both bound the same physical scratch allocation the selector sizes.
constexpr uint32_t kMaxScratchElems = 16384;

// Creates the Intel tile/split-K selection callback for the common subgroup-matrix
// kernel. Returns an empty selector on devices that are not Intel or that do not
// support the required 8x16x16 F16 subgroup matrix configuration, so the common
// factory yields no implementation and the caller falls back to the default
// MatMul path.
SubgroupMatrixTilingSelector CreateSubgroupMatrixTilingSelector(
    const ComputeContextBase& context);

// Like CreateSubgroupMatrixTilingSelector, but for the subgroup-matrix Conv path.
// The tile candidates are shared, but selection uses the Conv workgroup-memory
// budget (which also stages the im2col A tile) and the Conv-tuned tree instead
// of the MatMul table. Returns an empty selector on non-Intel devices.
SubgroupMatrixTilingSelector CreateSubgroupMatrixConvTilingSelector(
    const ComputeContextBase& context);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
