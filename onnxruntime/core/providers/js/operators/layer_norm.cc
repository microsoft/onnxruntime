// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "layer_norm.h"

namespace onnxruntime {
namespace js {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      LayerNormalization,                                             \
      kOnnxDomain,                                                    \
      17,                                                             \
      T,                                                              \
      kJsExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()), \
      LayerNorm<T, float>);

REGISTER_KERNEL_TYPED(float)
// REGISTER_KERNEL_TYPED(double)
// REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace js
}  // namespace onnxruntime
