// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quantization.h"
#include "attention_quantization_impl.cuh"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/common/safeint.h"
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
                                          const Tensor* past_tensor,
                                          void* parameters) const {
  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input->Shape(), weights->Shape(), bias->Shape(),
                                                 mask_index, past_tensor,
                                                 nullptr,  // relative_position_bias
                                                 parameters,
                                                 device_prop.maxThreadsPerBlock));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");

  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    if (0 != *(i_zp_tensor->Data<int8_t>()))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA only support symmetric quantization for Attention");
  }

  if (w_zp_tensor != nullptr) {
    // CUDA only support symmetric quantization for Attention
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    if (0 != *(w_zp_tensor->Data<int8_t>()))
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
  //   Input 5  - mask_index        : see Attention operator spec
  //   Input 6  - input_zero_point  : scalar
  //   Input 7  - weight_zero_point : scalar
  //   Input 8  - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0 - output            : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  AttentionParameters parameters;
  parameters.use_tf32 = UseTF32();

  ORT_RETURN_IF_ERROR(CheckInputs(input,
                                  weights,
                                  bias,
                                  input_scale_tensor,
                                  weight_scale_tensor,
                                  mask_index,
                                  i_zp_tensor,
                                  w_zp_tensor,
                                  past_tensor,
                                  &parameters));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;
  int hidden_size = parameters.hidden_size;
  int head_size = parameters.head_size;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = GetCublasHandle(context);
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = parameters.input_hidden_size;
  size_t num_elements = SafeInt<size_t>(m) * n;
  auto gemm_buffer = GetScratchBuffer<T>(num_elements * element_size, context->GetComputeStream());
  auto gemm_buffer_quantized = GetScratchBuffer<int32_t>(num_elements, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;

  ORT_RETURN_IF_ERROR(GemmInt8(m, n, k,
                               1 /*alpha_matmul*/, 0 /* beta_matmul*/,
                               input->Data<int8_t>(), k,
                               weights->Data<int8_t>(), n,
                               gemm_buffer_quantized.get(), n,
                               this,
                               context->GetComputeStream()));

  CudaT dequant_scale;
  CudaT input_scale = *(reinterpret_cast<const CudaT*>(input_scale_tensor->Data<T>()));
  CudaT weight_scale = *(reinterpret_cast<const CudaT*>(weight_scale_tensor->Data<T>()));
  if constexpr (sizeof(T) == 2) {
    dequant_scale = __float2half(__half2float(input_scale) * __half2float(weight_scale));
  } else {
    dequant_scale = input_scale * weight_scale;
  }

  // scale back and bias
  // TODO(tianleiwu): fuse Dequantize with Add bias and Transpose.
  ORT_RETURN_IF_ERROR(CudaDequantizeWithBias(Stream(context),
                                             gemm_buffer_quantized.get(),
                                             reinterpret_cast<const CudaT*>(bias->Data<T>()),
                                             reinterpret_cast<CudaT*>(gemm_buffer.get()),
                                             dequant_scale,
                                             m,
                                             n));

  std::vector<int64_t> present_dims{2, parameters.batch_size, parameters.num_heads,
                                    parameters.total_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);

  void* fused_runner = nullptr;  // TODO(tianleiwu): use fused kernel to speed up
  constexpr bool use_fused_cross_attention = false;
  constexpr bool use_memory_efficient_attention = false;
  constexpr bool use_flash_attention = false;
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   batch_size,
                                                   parameters.num_heads,
                                                   head_size,
                                                   parameters.v_head_size,
                                                   sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_flash_attention,
                                                   use_fused_cross_attention,
                                                   use_memory_efficient_attention);

  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  if (nullptr != mask_index) {
    data.mask_index = mask_index->Data<int>();
    data.mask_index_dims = mask_index->Shape().GetDims();
  }

  if (nullptr != past_tensor) {
    data.past = reinterpret_cast<const CudaT*>(past_tensor->Data<T>());
  }

  data.has_qkv_workspace = true;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  if (nullptr != present) {
    data.present = reinterpret_cast<CudaT*>(present->MutableData<T>());
  }

  return QkvToContext<CudaT>(GetDeviceProp(), cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
