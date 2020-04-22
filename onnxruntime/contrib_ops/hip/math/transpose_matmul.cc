// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      TransposeMatMul,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kHipExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::hip::MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime
