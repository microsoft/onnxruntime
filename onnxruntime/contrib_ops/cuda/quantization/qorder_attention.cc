// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qorder_common.h"
#include "qorder_common_impl.h"
#include "qorder_attention.h"
#include "qorder_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include <iostream>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QOrderedAttention,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", BuildKernelDefConstraints<float>())
        .TypeConstraint("G", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .InputMemoryType(OrtMemTypeCPUInput, 8),
    QOrderedAttention<MLFloat16>);

template <typename T>
QOrderedAttention<T>::QOrderedAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info) {
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight");
  order_bias_ = GetCublasLtOrderAttr(info, "order_bias");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  if (order_input_ == CUBLASLT_ORDER_ROW) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL, "Only CUBLASLT_ORDER_COL is supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_output");
  } else if (order_input_ == CUBLASLT_ORDER_COL32) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL4_4R2_8C || order_weight_ == CUBLASLT_ORDER_COL32_2R_4R4,
                "Only CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4 are supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_output");
  } else {
    ORT_ENFORCE(false, "Only CUBLASLT_ORDER_ROW or CUBLASLT_ORDER_COL32 are supported for order_input");
  }
}

template <typename T>
Status QOrderedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  // inputs are column based
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(4); // Support MLFloat16 in the future
  const Tensor* mask_index = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();

  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask_index, nullptr, nullptr, device_prop.maxThreadsPerBlock));

  const Tensor* scale_input = context->Input<Tensor>(1);
  const Tensor* scale_weights = context->Input<Tensor>(3);
  // const Tensor* scale_bias = context->Input<Tensor>(5);
  const Tensor* scale_gemm = context->Input<Tensor>(6);
  const Tensor* scale_output = context->Input<Tensor>(8);
  const Tensor* past = context->Input<Tensor>(9);

  const float* scale_input_data = scale_input->template Data<float>();
  const float* scale_weights_data = scale_weights->template Data<float>();
  // const float* scale_bias_data = scale_bias->template Data<float>();
  const float* scale_gemm_data = scale_gemm->template Data<float>();
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

  TensorShapeVector output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Past and Present will be row32 and fp16
  int past_sequence_length = 0;
  Tensor* present = GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length);

  cublasLtHandle_t cublasLt = CublasLtHandle();
  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer_quantized = GetScratchBuffer<int8_t>(2 * m * n);  // col32 + row

  cudaStream_t stream = Stream();

  // Gemm result(M, N) = scale_input * input * scale_weights * weights + scale_bias x B.
  const float scale_alpha = *scale_input_data * *scale_weights_data / *scale_gemm_data;
  ORT_RETURN_IF_ERROR(
      QOrdered_MatMul(cublasLt, stream, device_prop,
                      1, m, n, k,
                      &scale_alpha, input->template Data<int8_t>(), weights->template Data<int8_t>(),
                      bias->Data<float>(), gemm_buffer_quantized.get(),
                      (cublasLtOrder_t)order_weight_));

  typedef typename ToCudaType<T>::MappedType CudaT;
  constexpr size_t element_size = sizeof(T);

  auto gemm_buffer = GetScratchBuffer<int8_t>(m * n * element_size);  // row, fp16

  QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, GetDeviceProp(), gemm_buffer_quantized.get(), (CudaT*)gemm_buffer.get(),
                        *(const float*)scale_gemm_data, batch_size, sequence_length, n);

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, 0);
  auto temp_buffer = GetScratchBuffer<void>(workSpaceSize);
  auto output_buffer = GetScratchBuffer<int8_t>(m * n * element_size);  // row, fp16
  cublasHandle_t cublas = CublasHandle();
  if (!LaunchQOrderAttentionKernel(
          device_prop,
          stream,
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
          reinterpret_cast<CudaT*>(output_buffer.get()),
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
          nullptr,
          nullptr == present ? nullptr : present->template MutableData<T>())) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  QOrderQuantizeRowTo((cublasLtOrder_t)order_input_, stream, GetDeviceProp(),
                      (const CudaT*)output_buffer.get(), output->MutableData<int8_t>(),
                      *(const float*)scale_output_data, batch_size, sequence_length, hidden_size);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
