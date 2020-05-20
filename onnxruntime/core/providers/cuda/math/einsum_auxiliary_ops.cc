// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace cuda {

namespace EinsumOp {

std::unique_ptr<Tensor> Transpose(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<size_t>& permutation, AllocatorPtr allocator,
                                  const OpKernelInfo& info) {
  auto input_rank = input_shape_override.size();
  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutation must match the rank of the input to be permutated");

  std::vector<int64_t> output_dims;
  output_dims.reserve(input_rank);

  for (const auto& dim : permutation) {
    output_dims.push_back(input_shape_override.at(dim));
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = onnxruntime::make_unique<Tensor>(input.DataType(), output_dims, allocator);

  TensorShape overriden_shape(input_shape_override);
  cuda::Transpose::DoTranspose(cuda::Transpose(info), permutation, input, *output, &overriden_shape);

  return output;
}

/*
template <typename T>
std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
                               const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
                               AllocatorPtr allocator, concurrency::ThreadPool* tp) {
  // Sanity checks before the actual MatMul
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match for MatMul");
  ORT_ENFORCE(input_shape_1_override.size() == 3 && input_shape_2_override.size() == 3, "Only 1 batch dimension is allowed for MatMul");
  ORT_ENFORCE(input_shape_1_override[0] == input_shape_2_override[0], "Batch dimension should match for MatMul;");
  ORT_ENFORCE(input_shape_1_override[2] == input_shape_2_override[1], "Incompatible matrix dimensions for matMul");

  size_t batches = static_cast<size_t>(input_shape_1_override[0]);
  size_t M = static_cast<size_t>(input_shape_1_override[1]);
  size_t K = static_cast<size_t>(input_shape_1_override[2]);
  size_t N = static_cast<size_t>(input_shape_2_override[2]);

  size_t left_offset = M * K;
  size_t right_offset = K * N;
  size_t output_offset = M * N;

  std::vector<int64_t> output_dims;
  output_dims.reserve(3);
  output_dims.push_back(static_cast<int64_t>(batches));
  output_dims.push_back(static_cast<int64_t>(M));
  output_dims.push_back(static_cast<int64_t>(N));

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = onnxruntime::make_unique<Tensor>(input_1.DataType(), output_dims, allocator);

  const T* input_1_data = input_1.template Data<T>();
  const T* input_2_data = input_2.template Data<T>();
  T* output_data = output->template MutableData<T>();

  // Process each batch
  // TODO: Currently we parallelize a single MatMul operation, add logic to determine if
  // we can parallelizing on batches would be more optimal
  for (size_t i = 0; i < batches; ++i) {
    math::MatMul<T>(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        input_1_data + i * left_offset,
        input_2_data + i * right_offset,
        output_data + i * output_offset, tp);
  }

  return output;
}

template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator, concurrency::ThreadPool* tp) {
}

// A specific helper just for the Diagonal op
static inline bool IsTransposeRequiredForDiagonal(int64_t dim_1, int64_t dim_2, int64_t rank) {
}

template <typename T>
static void DiagonalDataAssignment(const T* input_data, T* output_data, int64_t batch_size, int64_t base_stride, int64_t inner_stride) {
}

std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator) {
}
*/

}  // namespace EinsumOp

}  // namespace cuda

}  // namespace onnxruntime
