// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define JSEP_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)              \
  ONNX_OPERATOR_KERNEL_EX(                                                         \
      OP_TYPE, kOnnxDomain, VERSION, kJsExecutionProvider,                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define JSEP_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                             \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kJsExecutionProvider,                      \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),               \
      KERNEL_CLASS);


JSEP_KERNEL_IMPL(Abs, Abs)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Abs, 1, 13, float, Abs)
JSEP_ELEMENTWISE_KERNEL(Abs, 14, float, Abs)

}  // namespace js
}  // namespace onnxruntime
