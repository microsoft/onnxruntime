// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define REG_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS)      \
  ONNX_OPERATOR_KERNEL_EX(                                          \
      OP_TYPE,                                                      \
      kOnnxDomain,                                                  \
      VERSION,                                                      \
      kJsExecutionProvider,                                         \
      KernelDefBuilder()                                            \
          .TypeConstraint("T",                                      \
                          {DataTypeImpl::GetTensorType<float>(),    \
                           DataTypeImpl::GetTensorType<int32_t>(),  \
                           DataTypeImpl::GetTensorType<uint32_t>(), \
                           DataTypeImpl::GetTensorType<bool>()}),   \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                      \
      OP_TYPE,                                                                            \
      kOnnxDomain,                                                                        \
      VERSION_FROM, VERSION_TO,                                                           \
      kJsExecutionProvider,                                                               \
      KernelDefBuilder()                                                                  \
          .TypeConstraint("T",                                                            \
                          {DataTypeImpl::GetTensorType<float>(),                          \
                           DataTypeImpl::GetTensorType<int32_t>(),                        \
                           DataTypeImpl::GetTensorType<uint32_t>(),                       \
                           DataTypeImpl::GetTensorType<bool>()}),                         \
      KERNEL_CLASS);

JSEP_KERNEL_IMPL(Where, Where)
REG_ELEMENTWISE_VERSIONED_KERNEL(Where, 9, 15, Where);
REG_ELEMENTWISE_KERNEL(Where, 16, Where);
}  // namespace js
}  // namespace onnxruntime
