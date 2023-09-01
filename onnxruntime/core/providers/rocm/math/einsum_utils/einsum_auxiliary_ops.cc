// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/tunable/gemm.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace EinsumOp {

namespace DeviceHelpers {

namespace RocmDeviceHelpers {

// ROCM EP specific Data copy helper
Status DataCopy(const Tensor& input, Tensor& output, void* einsum_rocm_assets) {
  ORT_ENFORCE(output.SizeInBytes() == input.SizeInBytes(),
              "Einsum op: The candidate output does not match the actual output's shape");
  // There are no string tensors in Einsum's case - so safely use memcpy
  HIP_RETURN_IF_ERROR(hipMemcpyAsync(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes(),
                                     hipMemcpyDeviceToDevice,
                                     static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->GetRocmStream()));

  return Status::OK();
}

// ROCM EP specific Transpose helper
Status Transpose(const gsl::span<const size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* einsum_rocm_assets) {
  return rocm::Transpose::DoTranspose(static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->rocm_ep_->GetDeviceProp(),
                                      static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->GetRocmStream(),
                                      static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->rocblas_handle_,
                                      permutation, input, output, input_shape_override);
}

// ROCM EP specific MatMul helper
template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* /*tp*/,
              void* einsum_rocm_assets) {
  typedef typename rocm::ToHipType<T>::MappedType HipT;

  namespace blas = rocm::tunable::blas;
  return blas::column_major::StridedBatchedGemm(
      static_cast<rocm::tunable::RocmTuningContext*>(
          static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->rocm_ep_->GetTuningContext()),
      static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->ort_stream_,
      static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->rocblas_handle_,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      N, M, K,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(input_2_data), N, right_stride,
      reinterpret_cast<const HipT*>(input_1_data), K, left_stride,
      /*beta=*/0.0f,
      reinterpret_cast<HipT*>(output_data), N, output_stride,
      num_batches);
}

// ROCM EP specific ReduceSum helper
template <typename T>
std::unique_ptr<Tensor> ReduceSum(const Tensor& input, gsl::span<const int64_t> reduce_axes,
                                  bool keep_dims, AllocatorPtr allocator,
                                  const TensorShape* input_shape_override,
                                  concurrency::ThreadPool* /*tp*/, void* einsum_rocm_assets) {
  return rocm::ReductionOps::ReduceCompute<T>(static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->gpu_allocator_, MIOPEN_REDUCE_TENSOR_ADD,
                                              allocator, input, reduce_axes,
                                              keep_dims, false, false, false,
                                              true, static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->ort_stream_,
                                              input_shape_override);
}

// ROCM EP specific Diagonal helper
std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator, void* einsum_rocm_assets) {
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

  // Make a copy - we are going to mutate the dims
  TensorShapeVector output_dims = input_shape.AsShapeVector();

  // Remove the dim value in `second_dim` -
  // The diagonal values are stored along `first_dim`
  output_dims.erase(output_dims.begin() + second_dim);

  auto output = Tensor::Create(input.DataType(), output_dims, allocator);

  TensorPitches input_strides(input.Shape().GetDims());
  rocm::TArray<int64_t> gpu_input_strides(input_strides);

  auto output_rank = static_cast<int32_t>(output_dims.size());
  rocm::TArray<rocm::fast_divmod> gpu_output_strides(output_rank);
  TensorPitches output_strides(output_dims);
  for (auto i = 0; i < output_rank; i++) {
    gpu_output_strides[i] = rocm::fast_divmod(static_cast<int>(output_strides[i]));
  }

  DiagonalImpl(
      static_cast<EinsumRocmAssets*>(einsum_rocm_assets)->GetRocmStream(),
      input.DataRaw(),
      input.Shape().GetDims().size(),
      first_dim,
      second_dim,
      gpu_input_strides,
      output->MutableDataRaw(),
      gpu_output_strides,
      TensorShape(output_dims).Size(),
      input.DataType()->Size());

  return output;
}

}  // namespace RocmDeviceHelpers

}  // namespace DeviceHelpers

// Explicit template instantiations of functions

// float
template Status DeviceHelpers::RocmDeviceHelpers::MatMul<float>(
    const float* input_1_data, const float* input_2_data, float* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_rocm_assets);

template std::unique_ptr<Tensor> DeviceHelpers::RocmDeviceHelpers::ReduceSum<float>(
    const Tensor& input, gsl::span<const int64_t> reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_rocm_assets);

// MLFloat16
template Status DeviceHelpers::RocmDeviceHelpers::MatMul<MLFloat16>(
    const MLFloat16* input_1_data, const MLFloat16* input_2_data, MLFloat16* output_data,
    size_t left_stride, size_t right_stride, size_t output_stride,
    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
    void* einsum_rocm_assets);

template std::unique_ptr<Tensor> DeviceHelpers::RocmDeviceHelpers::ReduceSum<MLFloat16>(
    const Tensor& input, gsl::span<const int64_t> reduce_axes,
    bool keep_dims, AllocatorPtr allocator,
    const TensorShape* input_shape_override,
    concurrency::ThreadPool* tp, void* einsum_rocm_assets);

}  // namespace EinsumOp

}  // namespace onnxruntime
