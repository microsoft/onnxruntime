// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "layer_norm.h"

namespace onnxruntime {
namespace js {

#define REGISTER_KERNEL_TYPED(T, U)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LayerNormalization,                                                                  \
      kOnnxDomain,                                                                         \
      17,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create())                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                           \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),                      \
      LayerNorm<T, U>);

REGISTER_KERNEL_TYPED(float, float)

}  // namespace js
}  // namespace onnxruntime
