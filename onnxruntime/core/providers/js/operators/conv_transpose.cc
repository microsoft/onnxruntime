// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

#include "conv_transpose.h"
namespace onnxruntime {
namespace js {
#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      ConvTranspose,                                                                       \
      kMSInternalNHWCDomain,                                                               \
      11,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T, true>);                                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T, false>);                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T, false>);

REGISTER_KERNEL_TYPED(float)

}  // namespace js
}  // namespace onnxruntime
