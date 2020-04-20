// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace EinsumOp {

Tensor Transpose(const Tensor& input, const std::vector<size_t>& permutation, AllocatorPtr allocator) {
  const auto& input_dims = input.Shape().GetDims();
  auto input_rank = input_dims.size();

  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutation must match the rank of the input to be permutated");

  std::vector<int64_t> output_dims;
  output_dims.reserve(input_rank);

  for (const auto& dim : permutation) {
    output_dims.push_back(input_dims.at(dim));
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  Tensor output(input.DataType(), output_dims, allocator);

  TransposeBase::DoTranspose(permutation, input, output);

  return output;
}

template <typename T>
Tensor MatMul(const Tensor& input_1, const Tensor& input_2, AllocatorPtr allocator, concurrency::ThreadPool* tp) {
  const auto& input1_dims = input_1.Shape().GetDims();
  const auto& input2_dims = input_2.Shape().GetDims();

  // Sanity checks before the actual MatMul
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match for MatMul");
  ORT_ENFORCE(input1_dims.size() == 3 && input2_dims.size() == 3, "Only 1 batch dimension is allowed for MatMul");
  ORT_ENFORCE(input1_dims[0] == input2_dims[0], "Batch dimension should match for MatMul;");
  ORT_ENFORCE(input1_dims[2] == input2_dims[1], "Incompatible matrix dimensions for matMul");

  size_t batches = static_cast<size_t>(input1_dims[0]);
  size_t M = static_cast<size_t>(input1_dims[1]);
  size_t K = static_cast<size_t>(input1_dims[2]);
  size_t N = static_cast<size_t>(input2_dims[2]);

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
  Tensor output(input_1.DataType(), output_dims, allocator);

  const T* input_1_data = input_1.template Data<T>();
  const T* input_2_data = input_2.template Data<T>();
  T* output_data = output.template MutableData<T>();

  // Process each batch
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
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes, 
                 AllocatorPtr allocator, concurrency::ThreadPool* tp) {
  return onnxruntime::ReduceSum<T>::Impl(input, reduce_axes, allocator, tp, true);
}

template <typename T>
Tensor ReduceSum(const Tensor& input, int64_t axis, AllocatorPtr allocator, concurrency::ThreadPool* tp) {
  std::vector<int64_t> reduce_axes(1, axis);
  return ReduceSum<T>(input, reduce_axes, allocator, tp);
}

// A specific helper just for the Diagonal op
static inline bool IsTransposeRequiredForDiagonal(int64_t dim_1, int64_t dim_2, int64_t rank) {
  // If the input is 2D, we don't need a transpose
  if (rank == 2)
    return false;

  // If the two dims are the innermost dims, no transpose is required
  if ((dim_1 == rank - 1 && dim_2 == rank - 2) ||
      (dim_1 == rank - 2 && dim_2 == rank - 1))
    return false;

  // Transpose is required
  return true;
}

// Parse diagnoal elements along the 2 innermost dimensions
// eg: input_shape = [1, 2, 3, 3]

// This implementation provides flexibility as to which of the 2 innermost dim values is preserved 
// via `preserve_innermost_dim_val` param

// preserve_innermost_dim_val == true,
//       output_shape = [1, 2, 1, 3] => the diagonal contains 3 elements and the dim value of the innermost dim is preserved

// preserve_innermost_dim_val == false,
//       output_shape = [1, 2, 3, 1] => the diagonal contains 3 elements and the dim value of the non-innermost dim is preserved
static Tensor DiagonalInnermostDims(const Tensor& input, bool preserve_innermost_dim_val, AllocatorPtr allocator) {
  const char* input_data = reinterpret_cast<const char*>(input.DataRaw());
  const auto& input_dims = input.Shape().GetDims();
  auto rank = input_dims.size();
  const size_t element_size_in_bytes = input.DataType()->Size();

  // This is an internal method and we already have finished all validations in the calling method.
  // We proceed without duplicating all validations again here.

  // We have a minimalistic check here to make sure the innermost dims have the same dim value
  // as the calling method may have done a transpose before calling this method
  ORT_ENFORCE(input_dims[rank - 2] == input_dims[rank - 1],
              "The innermost dims should have the same dim value to parse the diagonal elements");

  std::vector<int64_t> output_dims;
  output_dims.reserve(rank);

  int64_t num_iterations = 1;  // Flatten the outermost dims - this will be the number of iterations
  for (size_t i = 0; i < rank - 2; ++i) {
    auto input_dim_value = input_dims[i];
    num_iterations *= input_dim_value;
    output_dims.push_back(input_dim_value);
  }

  if (preserve_innermost_dim_val) {
    output_dims.push_back(1);
    output_dims.push_back(input_dims[rank - 1]);
  } else {
    output_dims.push_back(input_dims[rank - 1]);
    output_dims.push_back(1);
  }

  int64_t inner_stride = input_dims[rank - 1];        // offset to move over the innermost dim
  int64_t base_stride = inner_stride * inner_stride;  // offset to move over all the axes except the 2 innermost dims

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  Tensor output(input.DataType(), output_dims, allocator);
  char* output_data = reinterpret_cast<char*>(output.MutableDataRaw());

  int64_t output_iter = 0;
  // TODO: Parallelize this operation
  for (int64_t i = 0; i < num_iterations; ++i) {
    auto base_offset = i * base_stride;
    for (int64_t j = 0; j < inner_stride; ++j) {
      memcpy(output_data + output_iter * element_size_in_bytes,
             input_data + (base_offset + j * inner_stride + j) * element_size_in_bytes,
             element_size_in_bytes);
      output_iter++;
    }
  }

  return output;
}

Tensor Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator) {
  const auto& dims = input.Shape().GetDims();
  auto rank = static_cast<int64_t>(dims.size());

  ORT_ENFORCE(rank >= 2 && dim_1 >= 0 && dim_2 >= 0 && dim_1 < rank &&
                  dim_2 < rank && dim_1 != dim_2 && dims[dim_1] == dims[dim_2],
              "Cannot parse the diagonal elements along dims ", dim_1, " and ", dim_2, " for input shape ", input.Shape());

  int64_t first_dim = -1;  // first_dim holds the lesser of dim_1 and dim_2
  int64_t second_dim = -1; // second_dim holds the greater of dim_1 and dim_2
  if (dim_1 < dim_2) {
    first_dim = dim_1;
    second_dim = dim_2;
  } else {
    first_dim = dim_2;
    second_dim = dim_1;
  }

  Tensor output;
  bool preserve_innermost_dim_val = false;

  bool is_transpose_required = IsTransposeRequiredForDiagonal(dim_1, dim_2, rank);
  if (is_transpose_required) {
    std::vector<size_t> permutation(rank, 0);
    int64_t first_dim_axis = -1;  // This is the axis eventually occupied by the first_dim

    // If one of the diagonal dimensions is one of the 2 innermost dims, then leave it as such
    // so as to avoid transpose overhead
    if (first_dim == rank - 2) { // If rank - 2 is occupied by first_dim, keep it there
      permutation[rank - 2] = first_dim;
      first_dim_axis = rank - 2;
    } else {
      if (second_dim != rank - 2) {  // If rank - 2 is not occupied by second_dim, then put first_dim there
        permutation[rank - 2] = first_dim;
        first_dim_axis = rank - 2;
      } else {  // If rank - 2 is occupied by second_dim, then put first_dim in rank - 1
        permutation[rank - 1] = first_dim;
        first_dim_axis = rank - 1;
        preserve_innermost_dim_val = true; // We always want to preserve the dim value of the first_dim
      }
    }

    // Put the second_dim in the dim not occupied by the first_dim
    if (first_dim_axis != rank - 1) {
      permutation[rank - 1] = second_dim;
    } else {
      permutation[rank - 2] = second_dim;
    }

    int64_t iter = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != first_dim && i != second_dim) {
        permutation[iter++] = i;
      }
    }
    ORT_ENFORCE(iter == rank - 2);

    // Permuatate the input so that the dims from which we need the diagonal forms the innermost dims
    auto transposed = Transpose(input, permutation, allocator);

    // Parse the diagonal from the innermost dims
    output = DiagonalInnermostDims(transposed, preserve_innermost_dim_val, allocator);

    // Swap back the dimensions to the original axes ordering using a "reverse permutation"

    // Find the "reverse" permutation
    iter = 0;
    std::vector<size_t> reverse_permutation(rank, 0);
    for (const auto& perm : permutation) {
      reverse_permutation[perm] = iter++;
    }

    // Permutate using the reverse permutation to get back the original axes ordering
    output = Transpose(output, reverse_permutation, allocator);
  } else {
    // No transposing required
    output = DiagonalInnermostDims(input, preserve_innermost_dim_val, allocator);
  }

  // Make copy of the transposed output
  auto output_dims = output.Shape().GetDims();

  // Unsqueeze the reduced dim
  auto iter = output_dims.begin() + second_dim;
  output_dims.erase(iter);

  // Reshape to output_dims
  CreateReshapedView(output, output_dims);

  return output;
}

// Explicit template instantiation

// float
template Tensor MatMul<float>(const Tensor& input_1, const Tensor& input_2, 
    AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<float>(const Tensor& input, const std::vector<int64_t>& reduce_axes, 
    AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<float>(const Tensor& input, int64_t axis, AllocatorPtr allocator, concurrency::ThreadPool* tp);

// int32_t
template Tensor MatMul<int32_t>(const Tensor& input_1, const Tensor& input_2,
                              AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<int32_t>(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                 AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<int32_t>(const Tensor& input, int64_t axis, AllocatorPtr allocator, concurrency::ThreadPool* tp);

// double
template Tensor MatMul<double>(const Tensor& input_1, const Tensor& input_2,
                              AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<double>(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                 AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<double>(const Tensor& input, int64_t axis, AllocatorPtr allocator, concurrency::ThreadPool* tp);

// int64_t
template Tensor MatMul<int64_t>(const Tensor& input_1, const Tensor& input_2,
                              AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<int64_t>(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                 AllocatorPtr allocator, concurrency::ThreadPool* tp);
template Tensor ReduceSum<int64_t>(const Tensor& input, int64_t axis, AllocatorPtr allocator, concurrency::ThreadPool* tp);

}  // namespace EinsumOp
}  // namespace onnxruntime
