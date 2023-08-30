// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "gemm.h"

namespace onnxruntime {
namespace js {

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      9, 10,                                                                               \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);                                                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      7, 8,                                                                                \
      T,                                                                                   \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

REGISTER_KERNEL_TYPED(float)

}  // namespace js
}  // namespace onnxruntime
