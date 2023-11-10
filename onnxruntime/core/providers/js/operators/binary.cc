// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define REG_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)               \
  ONNX_OPERATOR_KERNEL_EX(                                                         \
      OP_TYPE,                                                                     \
      kOnnxDomain,                                                                 \
      VERSION,                                                                     \
      kJsExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                            \
      OP_TYPE,                                                                                  \
      kOnnxDomain,                                                                              \
      VERSION_FROM, VERSION_TO,                                                                 \
      kJsExecutionProvider,                                                                     \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),              \
      KERNEL_CLASS);

JSEP_KERNEL_IMPL(Add, Add)
REG_ELEMENTWISE_VERSIONED_KERNEL(Add, 7, 12, float, Add);
REG_ELEMENTWISE_VERSIONED_KERNEL(Add, 13, 13, float, Add);
REG_ELEMENTWISE_KERNEL(Add, 14, float, Add);

JSEP_KERNEL_IMPL(Sub, Sub)
REG_ELEMENTWISE_VERSIONED_KERNEL(Sub, 7, 12, float, Sub);
REG_ELEMENTWISE_VERSIONED_KERNEL(Sub, 13, 13, float, Sub);
REG_ELEMENTWISE_KERNEL(Sub, 14, float, Sub);

JSEP_KERNEL_IMPL(Mul, Mul)
REG_ELEMENTWISE_VERSIONED_KERNEL(Mul, 7, 12, float, Mul);
REG_ELEMENTWISE_VERSIONED_KERNEL(Mul, 13, 13, float, Mul);
REG_ELEMENTWISE_KERNEL(Mul, 14, float, Mul);

JSEP_KERNEL_IMPL(Div, Div)
REG_ELEMENTWISE_VERSIONED_KERNEL(Div, 7, 12, float, Div);
REG_ELEMENTWISE_VERSIONED_KERNEL(Div, 13, 13, float, Div);
REG_ELEMENTWISE_KERNEL(Div, 14, float, Div);

JSEP_KERNEL_IMPL(Pow, Pow)
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 7, 11, float, Pow);
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 12, 12, float, Pow);
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 13, 14, float, Pow);
REG_ELEMENTWISE_KERNEL(Pow, 15, float, Pow);

}  // namespace js
}  // namespace onnxruntime
