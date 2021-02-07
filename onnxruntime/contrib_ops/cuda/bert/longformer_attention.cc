// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "longformer_attention.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "longformer_attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LongformerAttention,                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LongformerAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LongformerAttention<T>::LongformerAttention(const OpKernelInfo& info) : CudaKernel(info), LongformerAttentionBase(info) {}

template <typename T>
Status LongformerAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention->Shape()));

  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Output 0 - output     : (batch_size, sequence_length, hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = CublasHandle();
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  size_t qkv_size = batch_size * sequence_length * 3 * hidden_size * element_size;
  auto gemm_buffer = GetScratchBuffer<T>(qkv_size);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  auto& device_prop = GetDeviceProp();
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
      GetConstOnes<CudaT>(m), 1,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // TODO: calculate the exact value from global flags.
  int max_num_global = sequence_length;

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run glboal GEMM.
  auto global_gemm_buffer = GetScratchBuffer<T>(max_num_global > 0 ? qkv_size : 0);

  if (max_num_global > 0) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
        reinterpret_cast<const CudaT*>(global_bias->template Data<T>()), n,
        GetConstOnes<CudaT>(m), 1,
        &zero, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));

    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(global_weights->template Data<T>()), n,
        reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
        &one, reinterpret_cast<CudaT*>(global_gemm_buffer.get()), n, device_prop));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, max_num_global);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchLongformerAttentionKernel(
          device_prop,
          Stream(),
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          reinterpret_cast<const CudaT*>(mask->template Data<T>()),
          reinterpret_cast<const CudaT*>(global_gemm_buffer.get()),
          global_attention->template Data<int>(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          window_,
          max_num_global,
          workspace_buffer.get(),
          cublas,
          element_size)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
