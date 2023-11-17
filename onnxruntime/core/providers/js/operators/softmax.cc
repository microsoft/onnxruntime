// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"

namespace onnxruntime {
namespace js {

#define REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(SoftmaxOp, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                       \
      SoftmaxOp,                                                                           \
      kOnnxDomain,                                                                         \
      sinceVersion, endVersion,                                                            \
      kJsExecutionProvider,                                                                \
      (*KernelDefBuilder::Create())                                                        \
          .TypeConstraint("T", JsepSupportedFloatTypes()),                                 \
      SoftmaxOp);

#define REGISTER_SOFTMAX_ELEMENTWISE_KERNEL(SoftmaxOp, sinceVersion) \
  ONNX_OPERATOR_KERNEL_EX(                                           \
      SoftmaxOp,                                                     \
      kOnnxDomain,                                                   \
      sinceVersion,                                                  \
      kJsExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", JsepSupportedFloatTypes())            \
          .InputMemoryType(OrtMemTypeCPU, 1),                        \
      SoftmaxOp);

REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(Softmax, 1, 10);
REGISTER_SOFTMAX_ELEMENTWISE_VERSIONED_KERNEL(Softmax, 11, 12);
REGISTER_SOFTMAX_ELEMENTWISE_KERNEL(Softmax, 13);

}  // namespace js
}  // namespace onnxruntime
