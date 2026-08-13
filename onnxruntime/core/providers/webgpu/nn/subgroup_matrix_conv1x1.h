// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Abstract base class for an optional optimized implementation of the 1x1 (or
// same-size) Conv matmul path. Mirrors the MatMul::MatMulOptImpl / Gemm::GemmOptImpl
// pattern so Conv-specific optimizations can be plugged in without touching the
// generic MatMul code.
class Conv1x1OptImpl {
 public:
  virtual ~Conv1x1OptImpl() = default;

  // Attempts the subgroup-matrix path for a 1x1 / same-size Conv. Reads input/weight/bias
  // from `context` (prepacked_kernel is the prepacked OHWI->HWIO weight, or null to read the
  // original weight from input 1), runs its own shape inference, decides whether the problem
  // qualifies (group == 1, 1x1 or same-size, no fused activation, supported dtype), builds the
  // matmul operands, allocates the output and dispatches. Sets handled=true when it ran; leaves
  // handled=false (allocating nothing) so the caller falls back to the normal Conv path.
  // w_is_constant tells it whether the weight is a constant initializer (so an odd-N weight can
  // be padded once and cached).
  virtual Status Compute(ComputeContext& context, const ConvAttributes& conv_attrs,
                         const Activation& activation, bool is_channels_last,
                         const Tensor* prepacked_kernel, bool w_is_constant,
                         /*out*/ bool& handled) = 0;
};

// Creates a subgroup-matrix Conv 1x1 implementation on devices whose vendor policy
// supports the subgroup-matrix kernel; returns nullptr otherwise, so the caller
// falls back to the generic MatMul path. Reuses the shared subgroup-matrix kernel
// program and tiling policy without modifying subgroup_matrix_matmul.cc.
std::unique_ptr<Conv1x1OptImpl> CreateSubgroupMatrixConv1x1Impl(const ComputeContextBase& context);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
