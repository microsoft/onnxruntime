// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts implementations and thin wrappers over other onnx operator implementations
// that will be called from within the Einsum operator implementation

#pragma once

#ifndef SHARED_PROVIDER
#include "core/util/math.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cpu/reduction/reduction_ops.h"
#endif

#include <vector>

namespace onnxruntime {

namespace EinsumOp {

// Holds device specific implementations of the following ops
// This is because the CPU kernel needs to call the CPU implementation
// and the CUDA kernel will call the CUDA implementation and share the rest of the code

namespace DeviceHelpers {

// Data copy op - Copies raw data from the source tensor's buffer to the destination tensor's buffer
using DataCopy = std::function<Status(const Tensor& input, Tensor& output, void* einsum_cuda_assets)>;

// Create tensor op - Creates an intermediate tensor
using CreateTensor = std::function<std::unique_ptr<Tensor>(const DataTypeImpl* type, const TensorShape& shape, AllocatorPtr allocator)>;

// Zero buffer op - Sets all bytes in the tensor's buffer to zero
using ZeroBuffer = std::function<Status(Tensor& input, void* einsum_cuda_assets)>;

// Transpose op - Transposes given input based on data in `permutation`
using Transpose = std::function<Status(const gsl::span<const size_t>& permutation, const Tensor& input,
                                       Tensor& output, const TensorShape* input_shape_override,
                                       void* einsum_cuda_assets)>;

// MatMul op - Multiplies two inputs of shapes [num_batches, M, K] and [num_batches, K, N]
template <typename T>
using MatMul = std::function<Status(const T* input_1_data, const T* input_2_data, T* output_data,
                                    size_t left_stride, size_t right_stride, size_t output_stride,
                                    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
                                    const void* mlas_backend_config,
                                    void* einsum_cuda_assets)>;

// ReduceSum op - Reduces along `reduce_axes`
template <typename T>
using ReduceSum = std::function<std::unique_ptr<Tensor>(const Tensor& input, gsl::span<const int64_t> reduce_axes,
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

std::unique_ptr<Tensor> CreateTensor(const DataTypeImpl* type, const TensorShape& shape, AllocatorPtr allocator);

Status ZeroBuffer(Tensor& input, void* einsum_cuda_assets);

Status Transpose(const gsl::span<const size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* einsum_cuda_assets);

template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
              const void* mlas_backend_config,
              void* einsum_cuda_assets);

template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, gsl::span<const int64_t> reduce_axes,
                                  bool keep_dims, AllocatorPtr allocator,
                                  const TensorShape* input_shape_override,
                                  concurrency::ThreadPool* tp, void* einsum_cuda_assets);

std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator, void* einsum_cuda_assets);

}  // namespace CpuDeviceHelpers

}  // namespace DeviceHelpers

// This helps decide if we need to apply (and pay the cost) of a Transpose
inline bool IsTransposeRequired(size_t input_rank, const gsl::span<const size_t>& permutation) {
  ORT_ENFORCE(input_rank == permutation.size(), "The rank of the input must match permutation size for Transpose");

  // No transpose required for scalars
  if (input_rank == 0) {
    return false;
  }

  // Weeds out cases where permutation is something like [0, 1, 2] for a 3D input and so on
  bool is_transpose_required = false;
  for (size_t i = 0; i < input_rank; ++i) {
    if (permutation[i] != i) {
      is_transpose_required = true;
      break;
    }
  }
  return is_transpose_required;
}

// Thin wrapper over the Transpose op to be called from Einsum that does some checks and invokes the device specific helper
inline std::unique_ptr<Tensor> Transpose(const Tensor& input, const TensorShape& input_shape_override,
                                         const gsl::span<const size_t>& permutation, AllocatorPtr allocator, void* einsum_cuda_assets,
                                         const DeviceHelpers::Transpose& device_transpose_func,
                                         const DeviceHelpers::CreateTensor& device_create_tensor_func) {
  auto input_rank = input_shape_override.NumDimensions();
  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutation must match the rank of the input to be permutated");

  TensorShapeVector output_dims;
  output_dims.reserve(input_rank);

  for (const auto& dim : permutation) {
    output_dims.push_back(input_shape_override[dim]);
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = device_create_tensor_func(input.DataType(), output_dims, allocator);

  TensorShape overridden_shape(input_shape_override);

  auto status = device_transpose_func(permutation, input, *output, &overridden_shape, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(common::ONNXRUNTIME, common::FAIL, "Einsum op: Transpose failed: ", status.ErrorMessage());
  }
  return output;
}

// Thin wrapper over the MatMul op to be called from Einsum that does some checks and invokes the device specific helper
// Not using the MatMulHelper for checks and to compute output dims as it adds a lot of checking overhead involving transposes of the inputs
// In our case, we have a more simplistic version which doesn't need to have those checks
template <typename T>
inline std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const gsl::span<const int64_t>& input_1_shape_override,
                                      const Tensor& input_2, const gsl::span<const int64_t>& input_2_shape_override,
                                      AllocatorPtr allocator, concurrency::ThreadPool* tp, const void* mlas_backend_config,
                                      void* einsum_cuda_assets,
                                      const DeviceHelpers::MatMul<T>& device_matmul_func,
                                      const DeviceHelpers::CreateTensor& device_create_tensor_func) {
  // Sanity checks before the actual MatMul
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match for MatMul");
  ORT_ENFORCE(input_1_shape_override.size() == 3 && input_2_shape_override.size() == 3, "Only 1 batch dimension is allowed for MatMul");
  ORT_ENFORCE(input_1_shape_override[0] == input_2_shape_override[0], "Batch dimension should match for MatMul;");
  ORT_ENFORCE(input_1_shape_override[2] == input_2_shape_override[1], "Incompatible matrix dimensions for matMul");

  size_t batches = static_cast<size_t>(input_1_shape_override[0]);
  size_t M = static_cast<size_t>(input_1_shape_override[1]);
  size_t K = static_cast<size_t>(input_1_shape_override[2]);
  size_t N = static_cast<size_t>(input_2_shape_override[2]);

  size_t left_offset = M * K;
  size_t right_offset = K * N;
  size_t output_offset = M * N;

  TensorShapeVector output_dims;
  output_dims.reserve(3);
  output_dims.push_back(static_cast<int64_t>(batches));
  output_dims.push_back(static_cast<int64_t>(M));
  output_dims.push_back(static_cast<int64_t>(N));

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = device_create_tensor_func(input_1.DataType(), output_dims, allocator);

  const T* input_1_data = input_1.Data<T>();
  const T* input_2_data = input_2.Data<T>();
  T* output_data = output->template MutableData<T>();

  auto status = device_matmul_func(input_1_data, input_2_data, output_data,
                                   left_offset, right_offset, output_offset, batches, M, K, N, tp, mlas_backend_config, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(common::ONNXRUNTIME, common::FAIL, "Einsum op: Exception during MatMul operation: ",
              status.ErrorMessage());
  }

  return output;
}

// Thin wrapper over the ReduceSum op
template <typename T>
inline std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const TensorShape& input_shape_override,
                                         gsl::span<const int64_t> reduce_axes, AllocatorPtr allocator,
                                         concurrency::ThreadPool* tp, void* einsum_cuda_assets,
                                         const DeviceHelpers::ReduceSum<T>& device_reduce_sum_func,
                                         const DeviceHelpers::CreateTensor& /*device_create_tensor_func*/) {
  return device_reduce_sum_func(input, reduce_axes, true, allocator, &input_shape_override, tp, einsum_cuda_assets);
}

}  // namespace EinsumOp

}  // namespace onnxruntime
