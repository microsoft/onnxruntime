// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define JSEP_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)        \
  ONNX_OPERATOR_KERNEL_EX(                                                         \
      OP_TYPE, kOnnxDomain, VERSION, kJsExecutionProvider,                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define JSEP_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS)          \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      OP_TYPE, kOnnxDomain, VERSION, kJsExecutionProvider,               \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()), \
      KERNEL_CLASS);

#define JSEP_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                       \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kJsExecutionProvider,                \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()),                   \
      KERNEL_CLASS);

#define JSEP_ELEMENTWISE_MULTI_TYPED_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS)             \
  ONNX_OPERATOR_KERNEL_EX(                                                              \
      OP_TYPE, kOnnxDomain, VERSION, kJsExecutionProvider,                              \
      KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),     \
                                              DataTypeImpl::GetTensorType<MLFloat16>(), \
                                              DataTypeImpl::GetTensorType<int32_t>()}), \
      KERNEL_CLASS);

#define JSEP_ELEMENTWISE_MULTI_TYPED_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                   \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kJsExecutionProvider,                            \
      KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),                    \
                                              DataTypeImpl::GetTensorType<MLFloat16>(),                \
                                              DataTypeImpl::GetTensorType<int32_t>()}),                \
      KERNEL_CLASS);
// math

JSEP_KERNEL_IMPL(Abs, Abs)
JSEP_ELEMENTWISE_MULTI_TYPED_VERSIONED_KERNEL(Abs, 6, 12, Abs)
JSEP_ELEMENTWISE_MULTI_TYPED_KERNEL(Abs, 13, Abs)

JSEP_KERNEL_IMPL(Neg, Neg)
JSEP_ELEMENTWISE_MULTI_TYPED_VERSIONED_KERNEL(Neg, 6, 12, Neg)
JSEP_ELEMENTWISE_MULTI_TYPED_KERNEL(Neg, 13, Neg)

JSEP_KERNEL_IMPL(Floor, Floor)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Floor, 6, 12, Floor)
JSEP_ELEMENTWISE_KERNEL(Floor, 13, Floor)

JSEP_KERNEL_IMPL(Ceil, Ceil)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Ceil, 6, 12, Ceil)
JSEP_ELEMENTWISE_KERNEL(Ceil, 13, Ceil)

JSEP_KERNEL_IMPL(Reciprocal, Reciprocal)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Reciprocal, 6, 12, Reciprocal)
JSEP_ELEMENTWISE_KERNEL(Reciprocal, 13, Reciprocal)

JSEP_KERNEL_IMPL(Sqrt, Sqrt)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Sqrt, 6, 12, Sqrt)
JSEP_ELEMENTWISE_KERNEL(Sqrt, 13, Sqrt)

JSEP_KERNEL_IMPL(Exp, Exp)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Exp, 6, 12, Exp)
JSEP_ELEMENTWISE_KERNEL(Exp, 13, Exp)

JSEP_KERNEL_IMPL(Erf, Erf)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Erf, 9, 12, Erf)
JSEP_ELEMENTWISE_KERNEL(Erf, 13, Erf)

JSEP_KERNEL_IMPL(Sigmoid, Sigmoid)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Sigmoid, 6, 12, Sigmoid)
JSEP_ELEMENTWISE_KERNEL(Sigmoid, 13, Sigmoid)

JSEP_KERNEL_IMPL(Log, Log)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Log, 6, 12, Log)
JSEP_ELEMENTWISE_KERNEL(Log, 13, Log)

JSEP_KERNEL_IMPL(Sin, Sin)
JSEP_ELEMENTWISE_KERNEL(Sin, 7, Sin)

JSEP_KERNEL_IMPL(Cos, Cos)
JSEP_ELEMENTWISE_KERNEL(Cos, 7, Cos)

JSEP_KERNEL_IMPL(Tan, Tan)
JSEP_ELEMENTWISE_KERNEL(Tan, 7, Tan)

JSEP_KERNEL_IMPL(Asin, Asin)
JSEP_ELEMENTWISE_KERNEL(Asin, 7, Asin)

JSEP_KERNEL_IMPL(Acos, Acos)
JSEP_ELEMENTWISE_KERNEL(Acos, 7, Acos)

JSEP_KERNEL_IMPL(Atan, Atan)
JSEP_ELEMENTWISE_KERNEL(Atan, 7, Atan)

JSEP_KERNEL_IMPL(Sinh, Sinh)
JSEP_ELEMENTWISE_KERNEL(Sinh, 9, Sinh)

JSEP_KERNEL_IMPL(Cosh, Cosh)
JSEP_ELEMENTWISE_KERNEL(Cosh, 9, Cosh)

JSEP_KERNEL_IMPL(Asinh, Asinh)
JSEP_ELEMENTWISE_KERNEL(Asinh, 9, Asinh)

JSEP_KERNEL_IMPL(Acosh, Acosh)
JSEP_ELEMENTWISE_KERNEL(Acosh, 9, Acosh)

JSEP_KERNEL_IMPL(Atanh, Atanh)
JSEP_ELEMENTWISE_KERNEL(Atanh, 9, Atanh)

JSEP_KERNEL_IMPL(Tanh, Tanh)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Tanh, 6, 12, Tanh)
JSEP_ELEMENTWISE_KERNEL(Tanh, 13, Tanh)

JSEP_KERNEL_IMPL(Not, Not)
JSEP_ELEMENTWISE_TYPED_KERNEL(Not, 1, bool, Not)

// activation

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_2_DEFAULT(ClipV10, ClipV10, min, 3.402823e+38f, max, -3.402823e+38f)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Clip, 6, 10, ClipV10)
JSEP_KERNEL_IMPL(Clip, Clip)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Clip, kOnnxDomain, 11, 11, kJsExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("T", JsepSupportedFloatTypes())
                                      .InputMemoryType(OrtMemTypeCPU, 1)
                                      .InputMemoryType(OrtMemTypeCPU, 2),
                                  Clip);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Clip, kOnnxDomain, 12, 12, kJsExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("T", JsepSupportedFloatTypes())
                                      .InputMemoryType(OrtMemTypeCPU, 1)
                                      .InputMemoryType(OrtMemTypeCPU, 2),
                                  Clip);
ONNX_OPERATOR_KERNEL_EX(Clip, kOnnxDomain, 13, kJsExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", JsepSupportedFloatTypes())
                            .InputMemoryType(OrtMemTypeCPU, 1)
                            .InputMemoryType(OrtMemTypeCPU, 2),
                        Clip);

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(Elu, Elu, alpha, 1.0)
JSEP_ELEMENTWISE_KERNEL(Elu, 6, Elu)

JSEP_KERNEL_IMPL(Relu, Relu)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Relu, 6, 12, Relu)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Relu, 13, 13, Relu)
JSEP_ELEMENTWISE_KERNEL(Relu, 14, Relu)

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(LeakyRelu, LeakyRelu, alpha, 0.01)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(LeakyRelu, 6, 15, LeakyRelu)
JSEP_ELEMENTWISE_KERNEL(LeakyRelu, 16, LeakyRelu)

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(ThresholdedRelu, ThresholdedRelu, alpha, 1.0)
JSEP_ELEMENTWISE_KERNEL(ThresholdedRelu, 10, ThresholdedRelu)

}  // namespace js
}  // namespace onnxruntime
