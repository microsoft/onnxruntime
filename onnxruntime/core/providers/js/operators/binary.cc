// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define REG_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS)          \
  ONNX_OPERATOR_KERNEL_EX(                                              \
      OP_TYPE,                                                          \
      kOnnxDomain,                                                      \
      VERSION,                                                          \
      kJsExecutionProvider,                                             \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedDataTypes()), \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                      \
      OP_TYPE,                                                                            \
      kOnnxDomain,                                                                        \
      VERSION_FROM, VERSION_TO,                                                           \
      kJsExecutionProvider,                                                               \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedDataTypes()),                   \
      KERNEL_CLASS);

JSEP_KERNEL_IMPL(Add, Add)
REG_ELEMENTWISE_VERSIONED_KERNEL(Add, 7, 12, Add);
REG_ELEMENTWISE_VERSIONED_KERNEL(Add, 13, 13, Add);
REG_ELEMENTWISE_KERNEL(Add, 14, Add);

JSEP_KERNEL_IMPL(Sub, Sub)
REG_ELEMENTWISE_VERSIONED_KERNEL(Sub, 7, 12, Sub);
REG_ELEMENTWISE_VERSIONED_KERNEL(Sub, 13, 13, Sub);
REG_ELEMENTWISE_KERNEL(Sub, 14, Sub);

JSEP_KERNEL_IMPL(Mul, Mul)
REG_ELEMENTWISE_VERSIONED_KERNEL(Mul, 7, 12, Mul);
REG_ELEMENTWISE_VERSIONED_KERNEL(Mul, 13, 13, Mul);
REG_ELEMENTWISE_KERNEL(Mul, 14, Mul);

JSEP_KERNEL_IMPL(Div, Div)
REG_ELEMENTWISE_VERSIONED_KERNEL(Div, 7, 12, Div);
REG_ELEMENTWISE_VERSIONED_KERNEL(Div, 13, 13, Div);
REG_ELEMENTWISE_KERNEL(Div, 14, Div);

JSEP_KERNEL_IMPL(Pow, Pow)
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 7, 11, Pow);
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 12, 12, Pow);
REG_ELEMENTWISE_VERSIONED_KERNEL(Pow, 13, 14, Pow);
REG_ELEMENTWISE_KERNEL(Pow, 15, Pow);

JSEP_KERNEL_IMPL(Equal, Equal)
REG_ELEMENTWISE_VERSIONED_KERNEL(Equal, 7, 10, Equal);
REG_ELEMENTWISE_VERSIONED_KERNEL(Equal, 11, 12, Equal);
REG_ELEMENTWISE_VERSIONED_KERNEL(Equal, 13, 18, Equal);
REG_ELEMENTWISE_KERNEL(Equal, 19, Equal);

JSEP_KERNEL_IMPL(Greater, Greater)
REG_ELEMENTWISE_VERSIONED_KERNEL(Greater, 7, 8, Greater);
REG_ELEMENTWISE_VERSIONED_KERNEL(Greater, 9, 12, Greater);
REG_ELEMENTWISE_KERNEL(Greater, 13, Greater);

JSEP_KERNEL_IMPL(GreaterOrEqual, GreaterOrEqual)
REG_ELEMENTWISE_VERSIONED_KERNEL(GreaterOrEqual, 12, 15, GreaterOrEqual);
REG_ELEMENTWISE_KERNEL(GreaterOrEqual, 16, GreaterOrEqual);

JSEP_KERNEL_IMPL(Less, Less)
REG_ELEMENTWISE_VERSIONED_KERNEL(Less, 7, 8, Less);
REG_ELEMENTWISE_VERSIONED_KERNEL(Less, 9, 12, Less);
REG_ELEMENTWISE_KERNEL(Less, 13, Less);

JSEP_KERNEL_IMPL(LessOrEqual, LessOrEqual)
REG_ELEMENTWISE_VERSIONED_KERNEL(LessOrEqual, 12, 15, LessOrEqual);
REG_ELEMENTWISE_KERNEL(LessOrEqual, 16, LessOrEqual);

}  // namespace js
}  // namespace onnxruntime
