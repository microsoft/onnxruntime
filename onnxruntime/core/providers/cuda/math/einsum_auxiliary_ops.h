// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts implementations and thin wrappers over other onnx operator implementations
// that will be called from within the Einsum operator implementation

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

#include <vector>

namespace onnxruntime {

namespace cuda {

namespace EinsumOp {

// Thin wrapper over the Transpose op
std::unique_ptr<Tensor> Transpose(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<size_t>& permutation, AllocatorPtr allocator, const OpKernelInfo& info);

// Thin wrapper over the MatMul op
// Not using the MatMulHelper to compute output dims as it adds a lot of checking overhead involving transposes of the inputs
// In our case, we have a more simplistic version which doesn't need to have those checks
template <typename T>
std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const std::vector<int64_t>& input_1_shape_override,
                               const Tensor& input_2, const std::vector<int64_t>& input_2_shape_override,
                               AllocatorPtr allocator, concurrency::ThreadPool* tp);

// Thin wrapper over the ReduceSum op
template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator, concurrency::ThreadPool* tp);

// Diagonal - A specialized implementation somewhat similar to Torch's Diagonal op
// but is specific enough to what is just required for the Einsum op.
// Expects the input to be atleast 2-D and 0 <= dim_1, dim_2 < rank.
// input_shape[dim_1] == input_shape[dim_2] and dim_1 cannot be same as dim_2.
// The rank of the output is 1 less than the rank of the input and the squeezed dim is the greater of dim_1 and dim_2.

// Eg. input_shape = [2, 3, 5, 3] and dim_1 = 1 and dim_2 = 3
// The output_shape will be [2, 3, 5] and dim_1 will contain the diagonal elements
std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator);

}  // namespace EinsumOp

}  // namespace cuda

}  // namespace onnxruntime
