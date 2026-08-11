// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>
#include <vector>

#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {

// Abstract base class for an optional optimized implementation of the 1x1 (or
// same-size) Conv matmul path. Mirrors the MatMul::MatMulOptImpl / Gemm::GemmOptImpl
// pattern so Conv-specific optimizations can be plugged in without touching the
// generic MatMul code.
class Conv1x1OptImpl {
 public:
  virtual ~Conv1x1OptImpl() = default;

  // Attempts to compute Y = inputs[0] @ inputs[1] (+ optional bias inputs[2]) for
  // the Conv 1x1 path. input_a_reshape / input_b_reshape carry the logical operand
  // layout the Conv folds its N,H,W,C tensors into (the physical tensors are
  // unchanged); output is the caller's pre-allocated Conv output used as a flat
  // buffer. w_is_constant tells the implementation whether the weight operand is a
  // constant initializer (so it may pad an odd-N weight once and cache it). Sets
  // handled=true when it ran the optimized kernel; leaves handled=false to let the
  // caller fall back to the generic MatMul path.
  virtual Status Compute(ComputeContext& context,
                         std::vector<const Tensor*>& inputs, Tensor* output,
                         const TensorShape& input_a_reshape,
                         const TensorShape& input_b_reshape,
                         bool w_is_constant,
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
