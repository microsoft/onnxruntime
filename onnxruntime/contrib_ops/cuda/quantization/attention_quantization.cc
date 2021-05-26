// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quantization.h"
#include "attention_quantization_impl.cuh"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, TQuant)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QAttention,                                                        \
      kMSDomain,                                                         \
      1,                                                                 \
      T##_##TQuant,                                                      \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 6)                        \
          .InputMemoryType(OrtMemTypeCPUInput, 7)                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<TQuant>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<TQuant>())   \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      QAttention<T, TQuant>);

REGISTER_KERNEL_TYPED(float, int8_t)
REGISTER_KERNEL_TYPED(MLFloat16, int8_t)

template <typename T>
Status QAttention<T, int8_t>::CheckInputs(const Tensor* input,
                                          const Tensor* weights,
                                          const Tensor* bias,
                                          const Tensor* input_scale_tensor,
                                          const Tensor* weight_scale_tensor,
                                          const Tensor*& mask_index,
                                          const Tensor* i_zp_tensor,
                                          const Tensor* w_zp_tensor,
                                          const Tensor* past_tensor) const {
  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask_index, past_tensor, device_prop.maxThreadsPerBlock));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");

  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    if (0 != *(i_zp_tensor->template Data<int8_t>()))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA only support symmetric quantization for Attention");
  }

  if (w_zp_tensor != nullptr) {
    // CUDA only support symmetric quantization for Attention
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    if (0 != *(w_zp_tensor->template Data<int8_t>()))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA only support symmetric quantization for Attention");
  }

  return Status::OK();
}

template <typename T>
Status QAttention<T, int8_t>::ComputeInternal(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input 0  - input             : (batch_size, sequence_length, input_hidden_size)
  //   Input 1  - weights           : (input_hidden_size, 3 * hidden_size)
  //   Input 2  - bias              : (3 * hidden_size)
  //   Input 3  - input_scale       : scalar
  //   Input 4  - weight_scale      : scalar
  //   Input 5  - mask_index        : nullptr, (batch_size), (2 * batch_size), (batch_size, 1), (1, 1) or (batch_size, past_sequence_length + sequence_length)
  //   Input 6  - input_zero_point  : scalar
  //   Input 7  - weight_zero_point : scalar
  //   Input 8  - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0 - output            : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(CheckInputs(input,
                                  weights,
                                  bias,
                                  input_scale_tensor,
                                  weight_scale_tensor,
                                  mask_index,
                                  i_zp_tensor,
                                  w_zp_tensor,
                                  past_tensor));

  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);

  const auto& bias_shape = bias->Shape();
  const int hidden_size = static_cast<int>(bias_shape.GetDims()[0]) / 3;
  const int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = CublasHandle();
  const size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(batch_size * sequence_length * 3 * hidden_size * element_size);
  auto gemm_buffer_quantized = GetScratchBuffer<int32_t>(batch_size * sequence_length * 3 * hidden_size);

  typedef typename ToCudaType<T>::MappedType CudaT;

  GemmInt8(m, n, k,
           1 /*alpha_matmul*/, 0 /* beta_matmul*/,
           input->template Data<int8_t>(), k,
           weights->template Data<int8_t>(), n,
           gemm_buffer_quantized.get(), n,
           this);

  CudaT dequant_scale;
  CudaT input_scale = *(reinterpret_cast<const CudaT*>(input_scale_tensor->template Data<T>()));
  CudaT weight_scale = *(reinterpret_cast<const CudaT*>(weight_scale_tensor->template Data<T>()));
  if (sizeof(T) == 2) {
    dequant_scale = __float2half(__half2float(input_scale) * __half2float(weight_scale));
  } else {
    dequant_scale = input_scale * weight_scale;
  }
  // scale back and bias
  CudaDequantizeWithBias(
      Stream(),
      gemm_buffer_quantized.get(),
      reinterpret_cast<const CudaT*>(bias->template Data<T>()),
      reinterpret_cast<CudaT*>(gemm_buffer.get()),
      dequant_scale,
      m,
      n);

  int past_sequence_length = 0;
  Tensor* present_tensor = GetPresent(context, past_tensor, batch_size, head_size, sequence_length, past_sequence_length);

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, past_sequence_length);
  auto temp_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchAttentionKernel(
          GetDeviceProp(),
          Stream(),
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? nullptr : &(mask_index->Shape().GetDims()),
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
          nullptr == past_tensor ? nullptr : past_tensor->template Data<T>(),
          nullptr == present_tensor ? nullptr : present_tensor->template MutableData<T>())) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
