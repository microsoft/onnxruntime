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

// math

JSEP_KERNEL_IMPL(Abs, Abs)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Abs, 6, 12, float, Abs)
JSEP_ELEMENTWISE_KERNEL(Abs, 13, float, Abs)

JSEP_KERNEL_IMPL(Neg, Neg)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Neg, 6, 12, float, Neg)
JSEP_ELEMENTWISE_KERNEL(Neg, 13, float, Neg)

JSEP_KERNEL_IMPL(Floor, Floor)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Floor, 6, 12, float, Floor)
JSEP_ELEMENTWISE_KERNEL(Floor, 13, float, Floor)

JSEP_KERNEL_IMPL(Ceil, Ceil)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Ceil, 6, 12, float, Ceil)
JSEP_ELEMENTWISE_KERNEL(Ceil, 13, float, Ceil)

JSEP_KERNEL_IMPL(Reciprocal, Reciprocal)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Reciprocal, 6, 12, float, Reciprocal)
JSEP_ELEMENTWISE_KERNEL(Reciprocal, 13, float, Reciprocal)

JSEP_KERNEL_IMPL(Sqrt, Sqrt)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Sqrt, 6, 12, float, Sqrt)
JSEP_ELEMENTWISE_KERNEL(Sqrt, 13, float, Sqrt)

JSEP_KERNEL_IMPL(Exp, Exp)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Exp, 6, 12, float, Exp)
JSEP_ELEMENTWISE_KERNEL(Exp, 13, float, Exp)

JSEP_KERNEL_IMPL(Erf, Erf)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Erf, 9, 12, float, Erf)
JSEP_ELEMENTWISE_KERNEL(Erf, 13, float, Erf)

JSEP_KERNEL_IMPL(Sigmoid, Sigmoid)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Sigmoid, 6, 12, float, Sigmoid)
JSEP_ELEMENTWISE_KERNEL(Sigmoid, 13, float, Sigmoid)

JSEP_KERNEL_IMPL(Sin, Sin)
JSEP_ELEMENTWISE_KERNEL(Sin, 7, float, Sin)

JSEP_KERNEL_IMPL(Cos, Cos)
JSEP_ELEMENTWISE_KERNEL(Cos, 7, float, Cos)

JSEP_KERNEL_IMPL(Tan, Tan)
JSEP_ELEMENTWISE_KERNEL(Tan, 7, float, Tan)

JSEP_KERNEL_IMPL(Asin, Asin)
JSEP_ELEMENTWISE_KERNEL(Asin, 7, float, Asin)

JSEP_KERNEL_IMPL(Acos, Acos)
JSEP_ELEMENTWISE_KERNEL(Acos, 7, float, Acos)

JSEP_KERNEL_IMPL(Atan, Atan)
JSEP_ELEMENTWISE_KERNEL(Atan, 7, float, Atan)

JSEP_KERNEL_IMPL(Sinh, Sinh)
JSEP_ELEMENTWISE_KERNEL(Sinh, 9, float, Sinh)

JSEP_KERNEL_IMPL(Cosh, Cosh)
JSEP_ELEMENTWISE_KERNEL(Cosh, 9, float, Cosh)

JSEP_KERNEL_IMPL(Asinh, Asinh)
JSEP_ELEMENTWISE_KERNEL(Asinh, 9, float, Asinh)

JSEP_KERNEL_IMPL(Acosh, Acosh)
JSEP_ELEMENTWISE_KERNEL(Acosh, 9, float, Acosh)

JSEP_KERNEL_IMPL(Atanh, Atanh)
JSEP_ELEMENTWISE_KERNEL(Atanh, 9, float, Atanh)

// activation

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_2_DEFAULT(ClipV10, ClipV10, min, 3.402823e+38f, max, -3.402823e+38f)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Clip, 6, 10, float, ClipV10)
JSEP_KERNEL_IMPL(Clip, Clip)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Clip, kOnnxDomain, 11, 11, kJsExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                      .InputMemoryType(OrtMemTypeCPU, 1)
                                      .InputMemoryType(OrtMemTypeCPU, 2),
                                  Clip);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Clip, kOnnxDomain, 12, 12, kJsExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                      .InputMemoryType(OrtMemTypeCPU, 1)
                                      .InputMemoryType(OrtMemTypeCPU, 2),
                                  Clip);
ONNX_OPERATOR_KERNEL_EX(Clip, kOnnxDomain, 13, kJsExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                            .InputMemoryType(OrtMemTypeCPU, 1)
                            .InputMemoryType(OrtMemTypeCPU, 2),
                        Clip);

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(Elu, Elu, alpha, 1.0)
JSEP_ELEMENTWISE_KERNEL(Elu, 6, float, Elu)

JSEP_KERNEL_IMPL(Relu, Relu)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Relu, 6, 12, float, Relu)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(Relu, 13, 13, float, Relu)
JSEP_ELEMENTWISE_KERNEL(Relu, 14, float, Relu)

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(LeakyRelu, LeakyRelu, alpha, 0.01)
JSEP_ELEMENTWISE_VERSIONED_KERNEL(LeakyRelu, 6, 15, float, LeakyRelu)
JSEP_ELEMENTWISE_KERNEL(LeakyRelu, 16, float, LeakyRelu)

JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(ThresholdedRelu, ThresholdedRelu, alpha, 1.0)
JSEP_ELEMENTWISE_KERNEL(ThresholdedRelu, 10, float, ThresholdedRelu)

}  // namespace js
}  // namespace onnxruntime
