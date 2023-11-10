// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

#include "conv.h"

namespace onnxruntime {
namespace js {

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      kMSInternalNHWCDomain,                                                               \
      11,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, true>);                                                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, false>);                                                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, false>);

REGISTER_KERNEL_TYPED(float)

}  // namespace js
}  // namespace onnxruntime
