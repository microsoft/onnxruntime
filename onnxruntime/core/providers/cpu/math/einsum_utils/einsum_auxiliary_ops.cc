// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_auxiliary_ops.h"

using namespace onnxruntime::common;

namespace onnxruntime {

namespace EinsumOp {

namespace DeviceHelpers {

namespace CpuDeviceHelpers {

// CPU specific Data copy helper
Status DataCopy(const Tensor& input, Tensor& output, void* /*einsum_cuda_assets*/) {
  ORT_ENFORCE(output.SizeInBytes() == input.SizeInBytes(),
              "Einsum op: The candidate output does not match the actual output's shape");
  // There are no string tensors in Einsum's case - so safely use memcpy
  memcpy(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes());
  return Status::OK();
}

// CPU specific Transpose helper
Status Transpose(const std::vector<size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* /*einsum_cuda_assets*/) {
  return TransposeBase::DoTranspose(permutation, input, output, input_shape_override);
}

// CPU specific MatMul helper
template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
              void* /*einsum_cuda_assets*/) {
  for (size_t i = 0; i < num_batches; ++i) {
    math::MatMul<T>(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        input_1_data + i * left_stride,
        input_2_data + i * right_stride,
        output_data + i * output_stride, tp);
  }

  return Status::OK();
}

// CPU specific ReduceSum helper
template <typename T>
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                 bool keep_dims, AllocatorPtr allocator,
                 const TensorShape* input_shape_override,
                 concurrency::ThreadPool* tp, void* /*einsum_cuda_assets*/) {
  return onnxruntime::ReduceSum<T>::Impl(input, reduce_axes,
                                         allocator, tp, keep_dims,
                                         input_shape_override);
}
// CPU specific Diagonal helper(s)
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

template <typename T>
static void DiagonalDataAssignment(const T* input_data, T* output_data, int64_t batch_size,
                                   int64_t base_stride, int64_t inner_stride) {
  int64_t output_iter = 0;
  // TODO: Parallelize this operation
  for (int64_t i = 0; i < batch_size; ++i) {
    auto base_offset = i * base_stride;
    for (int64_t j = 0; j < inner_stride; ++j) {
      output_data[output_iter] = input_data[base_offset + j * inner_stride + j];
      output_iter++;
    }
  }
}

// Parse diagonal elements along the 2 innermost dimensions
// E.g.: input_shape = [1, 2, 3, 3]

// This implementation provides flexibility as to which of the 2 innermost dim values is preserved
// via the `preserve_innermost_dim_val` parameter

// preserve_innermost_dim_val == true,
//       output_shape = [1, 2, 1, 3] => the diagonal contains 3 elements and the dim value of the innermost dim is preserved

// preserve_innermost_dim_val == false,
//       output_shape = [1, 2, 3, 1] => the diagonal contains 3 elements and the dim value of the non-innermost dim is preserved

static std::unique_ptr<Tensor> DiagonalInnermostDims(const Tensor& input,
                                                     bool preserve_innermost_dim_val, AllocatorPtr allocator) {
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

  int64_t batch_size = 1;  // Flatten the outermost dims - this will be the number of iterations
  for (size_t i = 0; i < rank - 2; ++i) {
    auto input_dim_value = input_dims[i];
    batch_size *= input_dim_value;
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
  std::unique_ptr<Tensor> output = onnxruntime::make_unique<Tensor>(input.DataType(), output_dims, allocator);

  switch (element_size_in_bytes) {
    case 4:
      DiagonalDataAssignment<float>(reinterpret_cast<const float*>(input.DataRaw()),
                                    reinterpret_cast<float*>(output->MutableDataRaw()),
                                    batch_size, base_stride, inner_stride);
      break;
    case 8:
      DiagonalDataAssignment<double>(reinterpret_cast<const double*>(input.DataRaw()),
                                     reinterpret_cast<double*>(output->MutableDataRaw()),
                                     batch_size, base_stride, inner_stride);
      break;

    default:
      ORT_THROW("Einsum op: Unsupported data type for Diagonal ", input.DataType());
  }

  return output;
}

std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator, void* /*einsum_cuda_assets*/) {
  const auto& input_shape = input.Shape();
  const auto& input_dims = input_shape.GetDims();
  auto rank = static_cast<int64_t>(input_dims.size());

  ORT_ENFORCE(rank >= 2 && dim_1 != dim_2 && input_dims[dim_1] == input_dims[dim_2],
              "Cannot parse the diagonal elements along dims ", dim_1, " and ", dim_2, " for input shape ", input_shape);

  int64_t first_dim = -1;   // first_dim holds the lesser of dim_1 and dim_2
  int64_t second_dim = -1;  // second_dim holds the greater of dim_1 and dim_2
  if (dim_1 < dim_2) {
    first_dim = dim_1;
    second_dim = dim_2;
  } else {
    first_dim = dim_2;
    second_dim = dim_1;
  }

  std::unique_ptr<Tensor> output;
  bool preserve_innermost_dim_val = false;

  bool is_transpose_required = IsTransposeRequiredForDiagonal(dim_1, dim_2, rank);
  if (is_transpose_required) {
    std::vector<size_t> permutation(rank, 0);
    int64_t first_dim_axis = -1;  // This is the axis eventually occupied by the first_dim

    // If one of the diagonal dimensions is one of the 2 innermost dims, then leave it as such
    // so as to avoid transpose overhead
    if (first_dim == rank - 2) {  // If rank - 2 is occupied by first_dim, keep it there
      permutation[rank - 2] = first_dim;
      first_dim_axis = rank - 2;
    } else {
      if (second_dim != rank - 2) {  // If rank - 2 is not occupied by second_dim, then put first_dim there
        permutation[rank - 2] = first_dim;
        first_dim_axis = rank - 2;
      } else {  // If rank - 2 is occupied by second_dim, then put first_dim in rank - 1
        permutation[rank - 1] = first_dim;
        first_dim_axis = rank - 1;
        preserve_innermost_dim_val = true;  // We always want to preserve the dim value of the first_dim
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

    // Permutate the input so that the dims from which we need the diagonal forms the innermost dims
    // (Pass in CPU Transpose function here as this Diagonal method will only be used for CPU based diagonal parsing)
    auto transposed = EinsumOp::Transpose(input, input_dims, permutation, allocator, nullptr, Transpose);

    // Parse the diagonal from the innermost dims
    output = DiagonalInnermostDims(*transposed, preserve_innermost_dim_val, allocator);

    // Swap back the dimensions to the original axes ordering using a "reverse permutation"

    // Find the "reverse" permutation
    iter = 0;
    std::vector<size_t> reverse_permutation(rank, 0);
    for (const auto& perm : permutation) {
      reverse_permutation[perm] = iter++;
    }

    // Permutate using the reverse permutation to get back the original axes ordering
    // (Pass in CPU Transpose function here as this Diagonal method will only be used for CPU based diagonal parsing)
    output = EinsumOp::Transpose(*output, output->Shape().GetDims(), reverse_permutation, allocator, nullptr, Transpose);
  } else {
    // No transposing required
    output = DiagonalInnermostDims(input, preserve_innermost_dim_val, allocator);
  }

  // Make copy of the output dims
  auto output_dims = output->Shape().GetDims();

  // Unsqueeze the reduced dim
  auto iter = output_dims.begin() + second_dim;
  output_dims.erase(iter);

  output->Reshape(output_dims);
  return output;
}

}  // namespace CpuDeviceHelpers

}  // namespace DeviceHelpers

// This helps decide if we need to apply (and pay the cost) of a Transpose
bool IsTransposeRequired(size_t input_rank, const std::vector<size_t>& permutation) {
  ORT_ENFORCE(input_rank == permutation.size(), "The rank of the input must match permutation size for Transpose");

  // No transpose required for scalars
  if (input_rank == 0) {
    return false;
  }

  // Weeds out cases where permutation is something like [0, 1, 2] for a 3D input and so on
  bool transpose_required = false;
  for (size_t i = 0; i < input_rank; ++i) {
    if (permutation[i] != i) {
      transpose_required = true;
      break;
    }
  }

  return transpose_required;
}

// The following are thin wrappers over device specific helpers
std::unique_ptr<Tensor> Transpose(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<size_t>& permutation, AllocatorPtr allocator,
                                  void* einsum_cuda_assets, const DeviceHelpers::Transpose& device_transpose_func) {
  auto input_rank = input_shape_override.size();
  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutation must match the rank of the input to be permutated");

  std::vector<int64_t> output_dims;
  output_dims.reserve(input_rank);

  for (const auto& dim : permutation) {
    output_dims.push_back(input_shape_override[dim]);
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = onnxruntime::make_unique<Tensor>(input.DataType(), output_dims, allocator);

  TensorShape overriden_shape(input_shape_override);

  auto status = device_transpose_func(permutation, input, *output, &overriden_shape, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Einsum op: Transpose failed: ", status.ErrorMessage());
  }
  return output;
}

template <typename T>
std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
                               const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
                               AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
                               const DeviceHelpers::MatMul<T>& device_matmul_func) {
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

  auto status = device_matmul_func(input_1_data, input_2_data, output_data,
                                   left_offset, right_offset, output_offset, batches, M, K, N, tp, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Einsum op: Exception during MatMul operation: ",
              status.ErrorMessage());
  }

  return output;
}

template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const std::vector<int64_t>& input_shape_override,
                                  const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
                                  concurrency::ThreadPool* tp, void* einsum_cuda_assets,
                                  const DeviceHelpers::ReduceSum<T>& device_reduce_sum_func) {
  TensorShape overriden_shape(input_shape_override);
  auto output = device_reduce_sum_func(input, reduce_axes, true, allocator, &overriden_shape, tp, einsum_cuda_assets);
  return onnxruntime::make_unique<Tensor>(std::move(output));
}

// Explicit template instantiations of functions

// float
template Status DeviceHelpers::CpuDeviceHelpers::MatMul<float>(
    const float* input_1_data, const float* input_2_data, float* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_cuda_assets);

template std::unique_ptr<Tensor> MatMul<float>(
    const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
    const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
    AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::MatMul<float>& device_matmul_func);

template Tensor DeviceHelpers::CpuDeviceHelpers::ReduceSum<float>(
    const Tensor& input, const std::vector<int64_t>& reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets);

template std::unique_ptr<Tensor> ReduceSum<float>(
    const Tensor& input, const std::vector<int64_t>& input_shape_override,
    const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets, const DeviceHelpers::ReduceSum<float>& device_reduce_sum_func);

// int32_t
template Status DeviceHelpers::CpuDeviceHelpers::MatMul<int32_t>(
    const int32_t* input_1_data, const int32_t* input_2_data, int32_t* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_cuda_assets);

template std::unique_ptr<Tensor> MatMul<int32_t>(
    const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
    const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
    AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::MatMul<int32_t>& device_matmul_func);

template Tensor DeviceHelpers::CpuDeviceHelpers::ReduceSum<int32_t>(
    const Tensor& input, const std::vector<int64_t>& reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets);

template std::unique_ptr<Tensor> ReduceSum<int32_t>(
    const Tensor& input, const std::vector<int64_t>& input_shape_override,
    const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::ReduceSum<int32_t>& device_reduce_sum_func);

// double
template Status DeviceHelpers::CpuDeviceHelpers::MatMul<double>(
    const double* input_1_data, const double* input_2_data, double* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_cuda_assets);

template std::unique_ptr<Tensor> MatMul<double>(
    const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
    const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
    AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::MatMul<double>& device_matmul_func);

template Tensor DeviceHelpers::CpuDeviceHelpers::ReduceSum<double>(
    const Tensor& input, const std::vector<int64_t>& reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets);

template std::unique_ptr<Tensor> ReduceSum<double>(
    const Tensor& input, const std::vector<int64_t>& input_shape_override,
    const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::ReduceSum<double>& device_reduce_sum_func);

// int64_t
template Status DeviceHelpers::CpuDeviceHelpers::MatMul<int64_t>(
    const int64_t* input_1_data, const int64_t* input_2_data, int64_t* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_cuda_assets);

template Tensor DeviceHelpers::CpuDeviceHelpers::ReduceSum<int64_t>(
    const Tensor& input, const std::vector<int64_t>& reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets);

template std::unique_ptr<Tensor> MatMul<int64_t>(
    const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
    const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
    AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::MatMul<int64_t>& device_matmul_func);

template std::unique_ptr<Tensor> ReduceSum<int64_t>(
    const Tensor& input, const std::vector<int64_t>& input_shape_override,
    const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets, const DeviceHelpers::ReduceSum<int64_t>& reduce_sum_func);

// MLFloat16
template std::unique_ptr<Tensor> MatMul<MLFloat16>(
    const Tensor& input_1, const std::vector<int64_t>& input_shape_1_override,
    const Tensor& input_2, const std::vector<int64_t>& input_shape_2_override,
    AllocatorPtr allocator, concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::MatMul<MLFloat16>& device_matmul_func);

template std::unique_ptr<Tensor> ReduceSum<MLFloat16>(
    const Tensor& input, const std::vector<int64_t>& input_shape_override,
    const std::vector<int64_t>& reduce_axes, AllocatorPtr allocator,
    concurrency::ThreadPool* tp, void* einsum_cuda_assets,
    const DeviceHelpers::ReduceSum<MLFloat16>& device_reduce_sum_func);

}  // namespace EinsumOp
}  // namespace onnxruntime
