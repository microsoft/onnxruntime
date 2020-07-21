// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      TransposeScaleMatMul,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::cuda::MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
