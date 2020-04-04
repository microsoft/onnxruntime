#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace EinsumOp {

Tensor Transpose(const Tensor& input, const std::vector<size_t>& permutation, const AllocatorPtr& allocator) {
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

void CreateReshapedView(Tensor& input, const std::vector<int64_t>& new_dims) {
  input.Reshape(new_dims);
}

template <typename T>
Tensor MatMul(const Tensor& input_1, const Tensor& input_2, const AllocatorPtr& allocator, concurrency::ThreadPool* tp) {
  const auto& input1_dims = input_1.Shape().GetDims();
  const auto& input2_dims = input_2.Shape().GetDims();

  // Sanity checks before the actual MatMul
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match");
  ORT_ENFORCE(input1_dims.size() == 3 && input2_dims.size() == 3, "Only 1 batch dimension is allowed");
  ORT_ENFORCE(input1_dims[0] == input2_dims[0], "Batch dimension should match");
  ORT_ENFORCE(input1_dims[2] == input2_dims[1], "Incompatible matrix dimensions for multiplication");

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
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes, const AllocatorPtr& allocator) {
  return onnxruntime::ReduceSum<T>::Impl(input, reduce_axes, allocator);
}

template <typename T>
Tensor ReduceSum(const Tensor& input, int64_t axis, const AllocatorPtr& allocator) {
  std::vector<int64_t> reduce_axes(1, axis);
  return ReduceSum<T>(input, reduce_axes, allocator);
}

// A specific helper just for the Diagonal op
static inline bool IsTransposeRequiredForDiagonal(int64_t dim_1, int64_t dim_2, int64_t rank) {
  // If the input is 2D, we don't need a transpose
  if (rank == 2)
    return false;

  // If the two dims are the innermost dims, no transpose is required
  if (dim_1 == rank - 1 && dim_2 == rank - 2)
    return false;

  if (dim_1 == rank - 2 && dim_2 == rank - 1)
    return false;

  return true;
}

// Parse diagnoal elements along the 2 innermost dimensions
// eg: input_shape = [1, 2, 3, 3]
// output_shape = [1, 2, 3, 1] => the diagonal contains 3 elements but the rank is preserved by adding an unsqueezed dim
template <typename T>
static Tensor DiagonalInnermostDims(const Tensor& input, const AllocatorPtr& allocator) {
  const T* input_data = input.template Data<T>();
  const auto& input_dims = input.Shape().GetDims();
  auto rank = input_dims.size();

  ORT_ENFORCE(input_dims[rank - 2] == input_dims[rank - 1],
              "The innermost dims should have the same dim value to parse the diagonal elements");

  std::vector<int64_t> output_dims;
  output_dims.reserve(rank);

  for (size_t i = 0; i < rank - 2; ++i) {
    output_dims.push_back(input_dims[i]);
  }
  output_dims.push_back(input_dims[rank - 1]);
  output_dims.push_back(1);

  int64_t inner_stride = input_dims[rank - 1];                        // offset to move over the innermost dim
  int64_t base_stride = input_dims[rank - 1] * input_dims[rank - 2];  // offset to move over all the axes except the 2 innermost dims

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  Tensor output(input.DataType(), output_dims, allocator);
  T* output_data = output.template MutableData<T>();

  int64_t output_iter = 0;
  // TODO: Parallelize if needed
  for (int64_t i = 0; i < base_stride; ++i) {
    for (int64_t j = 0; j < inner_stride; ++j) {
      output_data[output_iter++] = input_data[i * base_stride + j * inner_stride + j];
    }
  }

  return output;
}

template <typename T>
Tensor Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, const AllocatorPtr& allocator) {
  const auto& dims = input.Shape().GetDims();
  auto rank = dims.size();

  ORT_ENFORCE(rank >= 2 && dim_1 >= 0 && dim_2 >= 0 && dim_1 < static_cast<int64_t>(rank) &&
                  dim_2 < static_cast<int64_t>(rank) && dim_1 != dim_2 && dims[dim_1] == dims[dim_2],
              "Cannot parse the diagonal elements along dims ", dim_1, " and ", dim_2, " for input shape ", input.Shape());

  int64_t first_dim = -1;
  int64_t second_dim = -1;
  if (dim_1 < dim_2) {
    first_dim = dim_1;
    second_dim = dim_2;
  } else {
    first_dim = dim_2;
    second_dim = dim_1;
  }

  Tensor output;

  bool is_transpose_required = IsTransposeRequiredForDiagonal(dim_1, dim_2, static_cast<int64_t>(rank));

  if (is_transpose_required) {
    std::vector<size_t> permutation;
    permutation.reserve(rank);

    for (size_t i = 0; i < rank; ++i) {
      permutation.push_back(i);
    }

    permutation[first_dim] = rank - 2;
    permutation[rank - 2] = first_dim;

    permutation[second_dim] = rank - 1;
    permutation[rank - 1] = second_dim;

    // Permuatate the input so that the dims from which we need the diagonal forms the innermost dims
    auto transposed = Transpose(input, permutation, allocator);

    // Parse the diagonal from the innermost dims
    output = DiagonalInnermostDims<T>(transposed, allocator);

    // Swap back the dimensions to the same axes ordering
    output = Transpose(output, permutation, allocator);
  } else {
    // If transposing was never required, the diagonal is in the innermost dim - like it is required
    // so no transposing is required
    output = DiagonalInnermostDims<T>(input, allocator);
  }

  // Make copy of the transposed output
  auto output_dims = output.Shape().GetDims();

  // Remove the reduced dim
  auto iter = output_dims.begin() + second_dim;
  ORT_ENFORCE(*iter == 1);
  output_dims.erase(iter);

  // Reshape to output_dims
  CreateReshapedView(output, output_dims);

  return output;
}

// Explicit template instantiation
template Tensor MatMul<float>(const Tensor& input_1, const Tensor& input_2, const AllocatorPtr& allocator, concurrency::ThreadPool* tp);
template Tensor Diagonal<float>(const Tensor& input, int64_t dim_1, int64_t dim_2, const AllocatorPtr& allocator);
template Tensor ReduceSum<float>(const Tensor& input, const std::vector<int64_t>& reduce_axes, const AllocatorPtr& allocator);
template Tensor ReduceSum<float>(const Tensor& input, int64_t axis, const AllocatorPtr& allocator);

}  // namespace EinsumOp
}  // namespace onnxruntime
