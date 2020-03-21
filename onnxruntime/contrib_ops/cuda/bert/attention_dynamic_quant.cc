// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_dynamic_quant.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/matmul_integer.cuh"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/quantize_linear.h"
#include "attention_impl.h"
#include "attention_dynamic_quant_impl.cuh"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::QuantizeLinear;

#define REGISTER_KERNEL_TYPED_QL(T, U)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      QuantizeLinear,                                              \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      KernelDefBuilder()                                           \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<U>()), \
      QuantizeLinear<T, U>);

REGISTER_KERNEL_TYPED_QL(int8_t, MLFloat16)
REGISTER_KERNEL_TYPED_QL(uint8_t, MLFloat16)

#define REGISTER_KERNEL_TYPED(T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      AttentionDynamicQuant,                                       \
      kMSDomain,                                                   \
      1,                                                           \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      KernelDefBuilder()                                           \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      AttentionDynamicQuant<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
AttentionDynamicQuant<T>::AttentionDynamicQuant(const OpKernelInfo& info) : CudaKernel(info),
                                                                            AttentionBase(info) {
}

template <typename T>
Status AttentionDynamicQuant<T>::PadMatrix(
    int row,
    int col,
    int align_size,
    const int8_t*& src,
    int& pad_size,
    IAllocatorUniquePtr<int8_t>& temp_mem_holder) const {
  pad_size = align_size - col % align_size;
  if (pad_size != align_size) {
    temp_mem_holder = GetScratchBuffer<int8_t>(row * (col + pad_size));
    ORT_RETURN_IF_ERROR(PadMatrixInLeadingDimension(src, temp_mem_holder.get(), row, col, pad_size));
    src = temp_mem_holder.get();
  } else {
    pad_size = 0;
  }

  return Status::OK();
}

template <typename T>
Status AttentionDynamicQuant<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(CheckInputs(context));
  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Input 1 - weights     : (hidden_size, 3 * hidden_size)
  //   Input 2 - bias        : (3 * hidden_size)
  //   Input 3 - mask_index  : (batch_size)
  //   Output                : (batch_size, sequence_length, hidden_size)
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* weight_scale = context->Input<Tensor>(4);

  const auto dims = input->Shape().GetDims();
  int input_size = static_cast<int>(input->Shape().Size());
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

  typedef typename ToCudaType<T>::MappedType CudaT;

  // compute the AMax of input
  int max_idx;
  CudaT input_max;
  cublasAmaxHelper(cublas, input_size, reinterpret_cast<const CudaT*>(input->template Data<T>()), 1, &max_idx);
  CUDA_CALL(cudaMemcpy(&input_max,
                       input->template Data<T>() + max_idx,
                       sizeof(T),
                       cudaMemcpyDeviceToHost));

  CudaT r_scale;
  CudaT input_scale;
  if (sizeof(T) == 2) {
    r_scale = __float2half(127.f / __half2float(input_max));
    input_scale = __float2half(__half2float(input_max) / 127.f);
  } else {
    r_scale = 127.f / input_max;
    input_scale = input_max / 127.f;
  }

  // quantize input
  auto quantize_buffer = GetScratchBuffer<int8_t>(input_size);
  CudaQuantizeLinearSimple(reinterpret_cast<const CudaT*>(input->template Data<T>()),
                           quantize_buffer.get(),
                           r_scale,
                           input_size);

  // pad A and B to make their leading dimension be multiples of 32
  // because cublasGemmEx requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned
  const int align_size = 32;
  int a_pad_size = 0;
  int b_pad_size = 0;
  const int8_t* a_ptr = quantize_buffer.get();
  const int8_t* b_ptr = weights->template Data<int8_t>();
  IAllocatorUniquePtr<int8_t> a_padded;
  IAllocatorUniquePtr<int8_t> b_padded;
  ORT_RETURN_IF_ERROR(PadMatrix(m,
                                k,
                                align_size,
                                a_ptr,
                                a_pad_size,
                                a_padded));
  ORT_RETURN_IF_ERROR(PadMatrix(k,
                                n,
                                align_size,
                                b_ptr,
                                b_pad_size,
                                b_padded));

  int alpha = 1;
  int beta = 0;
  auto gemm_buffer_quantized = GetScratchBuffer<int32_t>(batch_size * sequence_length * 3 * hidden_size);
  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
      cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b_ptr,
      CUDA_R_8I,
      static_cast<int>(n + b_pad_size),
      a_ptr,
      CUDA_R_8I,
      static_cast<int>(k + a_pad_size),
      &beta,
      gemm_buffer_quantized.get(),
      CUDA_R_32I,
      n,
      CUDA_R_32I,
      CUBLAS_GEMM_DFALT));

  const CudaT* weight_scale_ptr = reinterpret_cast<const CudaT*>(weight_scale->template Data<T>());

  CudaT dequant_scale;
  if (sizeof(T) == 2) {
    dequant_scale = __float2half(__half2float(input_scale) * __half2float(*weight_scale_ptr));
  } else {
    dequant_scale = input_scale * (*weight_scale_ptr);
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
