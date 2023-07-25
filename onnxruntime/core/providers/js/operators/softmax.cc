// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"

namespace onnxruntime {
namespace js {

#define REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(SoftmaxOp, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      SoftmaxOp,                                                                          \
      kOnnxDomain,                                                                        \
      sinceVersion, endVersion,                                                           \
      float,                                                                              \
      kJsExecutionProvider,                                                               \
      (*KernelDefBuilder::Create())                                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),                     \
      SoftmaxOp<float>);

#define REGISTER_SOFTMAX_ELEMENTWISE_KERNEL(SoftmaxOp, sinceVersion)   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      SoftmaxOp,                                                      \
      kOnnxDomain,                                                      \
      sinceVersion,                                                     \
      float,                                                            \
      kJsExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())    \
          .InputMemoryType(OrtMemTypeCPU, 1),                           \
      SoftmaxOp<float>);


REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(Softmax, 1, 11);
REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(Softmax, 12, 12);
REGISTER_SOFTMAX_ELEMENTWISE_KERNEL(Softmax, 13);

}  // namespace js
}  // namespace onnxruntime
