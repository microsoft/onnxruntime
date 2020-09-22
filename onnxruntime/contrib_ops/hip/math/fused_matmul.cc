// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

#define REGISTER_KERNEL_TYPED(op_name, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op_name,                                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kHipExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::hip::MatMul<T>);

REGISTER_KERNEL_TYPED(TransposeMatMul, float)
REGISTER_KERNEL_TYPED(TransposeMatMul, double)
REGISTER_KERNEL_TYPED(TransposeMatMul, MLFloat16)

REGISTER_KERNEL_TYPED(FusedMatMul, float)
REGISTER_KERNEL_TYPED(FusedMatMul, double)
REGISTER_KERNEL_TYPED(FusedMatMul, MLFloat16)

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime
