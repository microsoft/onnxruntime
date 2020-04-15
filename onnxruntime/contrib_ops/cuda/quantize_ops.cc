// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/quantize_linear.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
