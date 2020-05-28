// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace EinsumOp {

namespace DeviceHelpers {

namespace CudaDeviceHelpers {

// CUDA EP specific Data copy helper
Status DataCopy(const Tensor& input, Tensor& output) {
  ORT_ENFORCE(output.SizeInBytes() == input.SizeInBytes(),
              "Einsum op: The candidate output does not match the actual output's shape");
  // There are no string tensors in Einsum's case - so safely use memcpy
  // TODO: Currently, triggers copy on stream 0, investigate if we can still do that *if* the kernel is launched in a different stream
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes(), cudaMemcpyDeviceToDevice));

  return Status::OK();
}

// CUDA EP specific Transpose helper
Status Transpose(const std::vector<size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* cublas_handle) {
  return cuda::Transpose::DoTranspose(reinterpret_cast<cublasHandle_t>(cublas_handle), permutation, input, output, input_shape_override);
}

// CUDA EP specific MatMul helper
template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* /*tp*/,
              void* cublas_handle) {
  typedef typename cuda::ToCudaType<T>::MappedType CudaT;

  CudaT one = cuda::ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = cuda::ToCudaType<T>::FromFloat(0.0f);

  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(reinterpret_cast<cublasHandle_t>(cublas_handle),
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        static_cast<int>(N),
                                                        static_cast<int>(M),
                                                        static_cast<int>(K),
                                                        &one,
                                                        reinterpret_cast<const CudaT*>(input_2_data),
                                                        static_cast<int>(N),
                                                        static_cast<int>(right_stride),
                                                        reinterpret_cast<const CudaT*>(input_1_data),
                                                        static_cast<int>(K),
                                                        static_cast<int>(left_stride),
                                                        &zero,
                                                        reinterpret_cast<CudaT*>(output_data),
                                                        static_cast<int>(N),
                                                        static_cast<int>(output_stride),
                                                        static_cast<int>(num_batches)));

  return Status::OK();
}

// CUDA EP specific ReduceSum helper
template <typename T>
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                 bool keep_dims, AllocatorPtr allocator,
                 const TensorShape* input_shape_override,
                 concurrency::ThreadPool* /*tp*/, void* cuda_ep) {
  return cuda::ReductionOps::ReduceCompute<T>(*reinterpret_cast<CUDAExecutionProvider*>(cuda_ep), CUDNN_REDUCE_TENSOR_ADD,
                                              allocator, input, reduce_axes,
                                              keep_dims, false, false, false,
                                              true, input_shape_override);
}

// CUDA EP specific Diagonal helper(s)
std::unique_ptr<Tensor> Diagonal(const Tensor& /*input*/, int64_t /*dim_1*/, int64_t /*dim_2*/, AllocatorPtr /*allocator*/) {
  ORT_THROW(ONNXRUNTIME, NOT_IMPLEMENTED, "Cuda Diagonal not implemented");
}

}  // namespace CudaDeviceHelpers

}  // namespace DeviceHelpers

// Explicit template instantiations of functions

// float
template Status DeviceHelpers::CudaDeviceHelpers::MatMul<float>(const float* input_1_data, const float* input_2_data, float* output_data,
                                                                size_t left_stride, size_t right_stride, size_t output_stride,
                                                                size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
                                                                void* cublas_handle);

template Tensor DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                                                                   bool keep_dims, AllocatorPtr allocator,
                                                                   const TensorShape* input_shape_override,
                                                                   concurrency::ThreadPool* tp, void* cuda_ep);

}  // namespace EinsumOp

}  // namespace onnxruntime
