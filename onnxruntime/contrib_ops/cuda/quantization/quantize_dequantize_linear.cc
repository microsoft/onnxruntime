// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_Q_KERNEL_TYPED(T, U)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      QuantizeLinear,                                              \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      QuantizeLinear<T, U>);

REGISTER_Q_KERNEL_TYPED(int8_t, MLFloat16)
REGISTER_Q_KERNEL_TYPED(uint8_t, MLFloat16)

#define REGISTER_DQ_KERNEL_TYPED(T, U)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      DequantizeLinear,                                            \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<U>()), \
      DequantizeLinear<T, U>);

REGISTER_DQ_KERNEL_TYPED(int8_t, MLFloat16)
REGISTER_DQ_KERNEL_TYPED(uint8_t, MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
