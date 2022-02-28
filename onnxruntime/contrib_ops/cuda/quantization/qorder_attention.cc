// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qorder_attention.h"
#include "attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QOrderedAttention,                                          \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QOrderedAttention<T>);

REGISTER_KERNEL_TYPED(int8_t)

template <typename T>
QOrderedAttention<T>::QOrderedAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info) {
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight");
  order_bias_ = GetCublasLtOrderAttr(info, "order_bias");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  //TODO: any checks?
}

template <typename T>
Status QOrderedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  // inputs are column based
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(6);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask_index, nullptr, nullptr, device_prop.maxThreadsPerBlock));

  const Tensor* scale_input = context->Input<Tensor>(1);
  const Tensor* scale_weights = context->Input<Tensor>(3);
  const Tensor* scale_bias = context->Input<Tensor>(5);
  const Tensor* scale_output = context->Input<Tensor>(7);

  const float* scale_input_data = scale_input->template Data<float>();
  const float* scale_weights_data = scale_weights->template Data<float>();
  const float* scale_bias_data = scale_bias->template Data<float>();
  const float* scale_output_data = scale_output->template Data<float>();

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);

  // bias shape (3 * hidden_size)
  const auto& bias_shape = bias->Shape();
  int hidden_size = static_cast<int>(bias_shape[0]) / 3;

  int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasLtHandle_t cublasLt = CublasLtHandle();
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(batch_size * sequence_length * 3 * hidden_size * element_size);

  // typedef typename ToCudaType<T>::MappedType CudaT;
  // CudaT one = ToCudaType<T>::FromFloat(1.0f);
  // CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  cudaStream_t stream = Stream();

  // Bias shape is (N), broadcast using scale_bias * B(M, N) = scale_bias * ones(M, 1) x bias(1, N) + 0 * B.
  CUBLAS_RETURN_IF_ERROR(
    QOrdered_Gemm(cublasLt, stream, (int)1, m, n, 1,
                  scale_bias, GetConstOnes<int8_t>(m), reinterpret_cast<const int8_t*>(bias->template Data<int8_t>()),
                  0.0f, reinterpret_cast<int8_t*>(gemm_buffer.get()),
                  order_bias_, order_bias_, order_bias_, device_prop));

  // Gemm result(M, N) = scale_input * input * scale_weights * weights + scale_bias x B.
  CUBLAS_RETURN_IF_ERROR(
    QOrdered_Gemm(cublasLt, stream, (int)1, m, n, k, scale_alpha,
                  reinterpret_cast<const int8_t*>(input->template Data<int8_t>()),
                  reinterpret_cast<const int8_t*>(weights->template Data<int8_t>()),
                  0.0f, reinterpret_cast<int8_t*>(gemm_buffer.get()),
                  order_input_, order_weights_, order_bias_, device_prop));

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, past_sequence_length);
  auto temp_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchAttentionKernel(
          device_prop,
          stream,
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          temp_buffer.get(),
          cublas,
          element_size,
          is_unidirectional_,
          past_sequence_length,
          nullptr == past ? nullptr : past->template Data<T>(),
          nullptr == extra_add_qk ? nullptr : extra_add_qk->template Data<T>(),
          nullptr == present ? nullptr : present->template MutableData<T>())) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
