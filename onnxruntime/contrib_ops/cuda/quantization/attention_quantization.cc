// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quantization.h"
#include "attention_quantization_impl.cuh"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/matmul_integer.cuh"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

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
      KernelDefBuilder()                                                 \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                        \
          .InputMemoryType<OrtMemTypeCPUInput>(5)                        \
          .InputMemoryType<OrtMemTypeCPUInput>(6)                        \
          .InputMemoryType<OrtMemTypeCPUInput>(7)                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<TQuant>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<TQuant>())   \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      QAttention<T, TQuant>);

REGISTER_KERNEL_TYPED(float, int8_t)
REGISTER_KERNEL_TYPED(MLFloat16, int8_t)

template <typename T>
Status QAttention<T, int8_t>::ComputeInternal(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input 0 - input             : (batch_size, sequence_length, hidden_size)
  //   Input 1 - weights           : (hidden_size, 3 * hidden_size)
  //   Input 2 - bias              : (3 * hidden_size)
  //   Input 3 - mask_index        : (batch_size)
  //   Input 4 - input_scale       : scalar
  //   Input 5 - weight_scale      : scalar
  //   Input 6 - input_zero_point  : scalar
  //   Input 7 - weight_zero_point : scalar
  //   Output                      : (batch_size, sequence_length, hidden_size)
  ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* input_scale_tensor = context->Input<Tensor>(4);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(5);
  /*const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);*/

  const auto dims = input->Shape().GetDims();
  /*int input_size = static_cast<int>(input->Shape().Size());*/
  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  int hidden_size = static_cast<int>(dims[2]);
  int head_size = hidden_size / num_heads_;

  TensorShape output_shape(dims);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = CublasHandle();
  const size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;
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
      gemm_buffer_quantized.get(),
      reinterpret_cast<const CudaT*>(bias->template Data<T>()),
      reinterpret_cast<CudaT*>(gemm_buffer.get()),
      dequant_scale,
      m,
      n);

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length);
  auto temp_buffer = GetScratchBuffer<void>(workSpaceSize);
  if (!LaunchAttentionKernel(
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          mask_index->template Data<int>(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          temp_buffer.get(),
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
