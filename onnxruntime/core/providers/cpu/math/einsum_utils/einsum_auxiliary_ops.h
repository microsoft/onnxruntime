// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts implementations and thin wrappers over other onnx operator implementations
// that will be called from within the Einsum operator implementation

#pragma once

#include "core/util/math.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

#include <vector>

namespace onnxruntime {

namespace EinsumOp {

// Holds device specific implementations of the following ops
// This is because the CPU kernel needs to call the CPU implementation
// and the CUDA kernel will call the CUDA implementation and share the rest of the code

namespace DeviceHelpers {

// Data copy op - Copies raw data from the source tensor's buffer to the destination tensor's buffer
using DataCopy = std::function<Status(const Tensor& input, Tensor& output, void* einsum_cuda_assets)>;

// Transpose op - Transposes given input based on data in `permutation`
using Transpose = std::function<Status(const std::vector<size_t>& permutation, const Tensor& input,
                                       Tensor& output, const TensorShape* input_shape_override,
                                       void* einsum_cuda_assets)>;

// MatMul op - Multiplies two inputs of shapes [num_batches, M, K] and [num_batches, K, N]
template <typename T>
using MatMul = std::function<Status(const T* input_1_data, const T* input_2_data, T* output_data,
                                    size_t left_stride, size_t right_stride, size_t output_stride,
                                    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
                                    void* einsum_cuda_assets)>;

// ReduceSum op - Reduces along `reduce_axes`
template <typename T>
using ReduceSum = std::function<Tensor(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                       bool keep_dims, AllocatorPtr allocator,
                                       const TensorShape* input_shape_override,
                                       concurrency::ThreadPool* tp, void* einsum_cuda_assets)>;

// Diagonal op
// Diagonal - A specialized implementation somewhat similar to Torch's Diagonal op
// but is specific enough to what is just required for the Einsum op.
// Expects the input to be atleast 2-D and 0 <= dim_1, dim_2 < rank.
// input_shape[dim_1] == input_shape[dim_2] and dim_1 cannot be same as dim_2.
// The rank of the output is 1 less than the rank of the input and the squeezed dim is the greater of dim_1 and dim_2.

// Eg. input_shape = [2, 3, 5, 3] and dim_1 = 1 and dim_2 = 3
// The output_shape will be [2, 3, 5] and dim_1 will contain the diagonal elements
using Diagonal = std::function<std::unique_ptr<Tensor>(const Tensor& input, int64_t dim_1, int64_t dim_2,
                                                       AllocatorPtr allocator, void* einsum_cuda_assets)>;

// These are CPU specific device helper implementations
namespace CpuDeviceHelpers {

Status DataCopy(const Tensor& input, Tensor& output, void* einsum_cuda_assets);

Status Transpose(const std::vector<size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* einsum_cuda_assets);

template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
              void* einsum_cuda_assets);

template <typename T>
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                 bool keep_dims, AllocatorPtr allocator,
                 const TensorShape* input_shape_override,
                 concurrency::ThreadPool* tp, void* einsum_cuda_assets);

std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator, void* einsum_cuda_assets);

}  // namespace CpuDeviceHelpers

}  // namespace DeviceHelpers

// This helps decide if we need to apply (and pay the cost) of a Transpose
bool IsTransposeRequired(size_t input_rank, const std::vector<size_t>& permutation);

// Thin wrapper over the Transpose op to be called from Einsum that does some checks and invokes the device specific helper
std::unique_ptr<Tensor> Transpose(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<size_t>& permutation, AllocatorPtr allocator, void* einsum_cuda_assets,
                                  const DeviceHelpers::Transpose& device_transpose_func);

// Thin wrapper over the MatMul op to be called from Einsum that does some checks and invokes the device specific helper
// Not using the MatMulHelper for checks and to compute output dims as it adds a lot of checking overhead involving transposes of the inputs
// In our case, we have a more simplistic version which doesn't need to have those checks
template <typename T>
std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const std::vector<int64_t>& input_1_shape_override,
                               const Tensor& input_2, const std::vector<int64_t>& input_2_shape_override,
                               AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
                               const DeviceHelpers::MatMul<T>& device_matmul_func);

// Thin wrapper over the ReduceSum op
template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
                                  concurrency::ThreadPool* tp, void* cuda_ep,
                                  const DeviceHelpers::ReduceSum<T>& device_reduce_sum_func);

}  // namespace EinsumOp

}  // namespace onnxruntime
