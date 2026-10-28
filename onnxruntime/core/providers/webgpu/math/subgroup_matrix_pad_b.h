// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>

#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {

// Pads a row-major f16 weight B [..., K, N] along its last dim to N_b (>= N),
// zero-filling the new columns, giving B an aligned row stride/offset for a
// subgroup-matrix row-major load. Treats B as a flat [rows, N] -> [rows, N_b] copy
// with rows = numel / N, so a 2D [K, N] weight and a batched [batch, K, N] one (or,
// for Conv, an HWIO [kh, kw, Cin, Cout] one) all work. Always runs the GPU pad
// pass; callers needing to pay this once for a constant weight should cache the
// result themselves (e.g. via std::call_once), as subgroup_matrix_matmul.cc does.
Status PadSubgroupMatrixWeightColumns(ComputeContext& context, const Tensor& b, const TensorShape& b_shape,
                                      uint32_t N, uint32_t n_b, std::unique_ptr<Tensor>& padded);

}  // namespace webgpu
}  // namespace onnxruntime
