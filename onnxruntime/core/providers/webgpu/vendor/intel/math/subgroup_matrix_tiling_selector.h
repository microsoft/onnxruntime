// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Creates the Intel tile/split-K selection callback for the common subgroup-matrix
// kernel. Returns an empty selector on devices that are not Intel or that do not
// support the required 8x16x16 F16 subgroup matrix configuration, so the common
// factory yields no implementation and the caller falls back to the default
// MatMul path.
SubgroupMatrixTilingSelector CreateSubgroupMatrixTilingSelector(
    const ComputeContextBase& context);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
