// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts implementations and thin wrappers over other onnx operator implementations
// that will be called from within the Einsum operator implementation

#pragma once

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/util/math.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

#include <vector>

namespace onnxruntime {

namespace EinsumOp {

// Thin wrapper over the Transpose op
Tensor Transpose(const Tensor& input, const std::vector<size_t>& permutation, const AllocatorPtr& allocator);

// Creates a "reshaped view" for the same tensor (i.e.) mutates the shape for the same tensor
// We will use it to introduce some "unsqueezed" dims (i.e.) extra dims with dim value as 1
inline void CreateReshapedView(Tensor& input, const std::vector<int64_t>& new_dims);

// Thin wrapper over the MatMul op
// Not using the MatMulHelper to compute output dims as it adds a lot of checking overhead involving transposes of the inputs
// In our case, we have a more simplistic version which doesn't need to have those checks
template <typename T>
Tensor MatMul(const Tensor& input_1, const Tensor& input_2, const AllocatorPtr& allocator, concurrency::ThreadPool* tp);

// Thin wrapper over the ReduceSum op
template <typename T>
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes, const AllocatorPtr& allocator);

// Thin wrapper over the ReduceSum op (overload)
template <typename T>
Tensor ReduceSum(const Tensor& input, int64_t axis, const AllocatorPtr& allocator);

// Diagonal - A specialized implementation somewhat similar to Torch's Diagonal op
// but is specific enough to what is required for the Einsum op.
// Expects the input to be atleast 2-D and 0 <= dim_1, dim_2 < rank.
// input_shape[dim_1] == input_shape[dim_2] and dim_1 cannot be same as dim_2.
// The rank of the output is 1 less than the rank of the input and the reduced dim is the higer of dim_1 and dim_2.

// Eg. input_shape = [2, 3, 5, 3] and dim_1 = 1 and dim_2 = 3
// The output_shape will be [2, 3, 5] and dim_1 will contain the diagonal elements in the original tensor along the specified dimensions
Tensor Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, const AllocatorPtr& allocator);

}  // namespace EinsumOp

}  // namespace onnxruntime
