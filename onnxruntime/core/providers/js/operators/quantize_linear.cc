// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"

namespace onnxruntime {
namespace js {
#define REGISTER_DEQUANTIZED_LINEAR_VERSIONED_TYPED_KERNEL(T, sinceVersion, endVerion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                             \
      DequantizeLinear,                                                                \
      kOnnxDomain,                                                                     \
      sinceVersion, endVerion,                                                         \
      T,                                                                               \
      kJsExecutionProvider,                                                            \
      (*KernelDefBuilder::Create())                                                    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())                      \
          .TypeConstraint("T2", JsepSupportedFloatTypes()),                            \
      DequantizeLinear);

#define REGISTER_DEQUANTIZED_LINEAR_TYPED_KERNEL(T, sinceVersion) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DequantizeLinear,                                           \
      kOnnxDomain,                                                \
      sinceVersion,                                               \
      T,                                                          \
      kJsExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", JsepSupportedFloatTypes()),       \
      DequantizeLinear);

#define REGISTER_DEQUANTIZED_LINEAR_VERSIONED_TYPED_KERNEL_PRE_19(T, sinceVersion, endVerion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                    \
      DequantizeLinear,                                                                       \
      kOnnxDomain,                                                                            \
      sinceVersion, endVerion,                                                                \
      T,                                                                                      \
      kJsExecutionProvider,                                                                   \
      (*KernelDefBuilder::Create())                                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                             \
      DequantizeLinear);

#define REGISTER_DEQUANTIZED_LINEAR_KERNEL_TYPED(T)                    \
  REGISTER_DEQUANTIZED_LINEAR_VERSIONED_TYPED_KERNEL_PRE_19(T, 10, 12) \
  REGISTER_DEQUANTIZED_LINEAR_VERSIONED_TYPED_KERNEL_PRE_19(T, 13, 18) \
  REGISTER_DEQUANTIZED_LINEAR_VERSIONED_TYPED_KERNEL(T, 19, 20)        \
  REGISTER_DEQUANTIZED_LINEAR_TYPED_KERNEL(T, 21)

REGISTER_DEQUANTIZED_LINEAR_KERNEL_TYPED(int8_t)
REGISTER_DEQUANTIZED_LINEAR_KERNEL_TYPED(uint8_t)
REGISTER_DEQUANTIZED_LINEAR_KERNEL_TYPED(int32_t)

}  // namespace js
}  // namespace onnxruntime
