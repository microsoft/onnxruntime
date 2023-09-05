// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "argminmax.h"

namespace onnxruntime {
namespace js {

#define REGISTER_ARGMAX_ELEMENTWISE_VERSIONED_KERNEL(ArgMinMaxOp, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                        \
      ArgMinMaxOp,                                                                          \
      kOnnxDomain,                                                                          \
      sinceVersion, endVersion,                                                             \
      kJsExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create())                                                         \
          .TypeConstraint("T", JsepSupportedFloatTypes()),                                  \
      ArgMinMaxOp<>);

#define REGISTER_ARGMAX_ELEMENTWISE_KERNEL(ArgMinMaxOp, sinceVersion) \
  ONNX_OPERATOR_KERNEL_EX(                                            \
      ArgMinMaxOp,                                                    \
      kOnnxDomain,                                                    \
      sinceVersion,                                                   \
      kJsExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", JsepSupportedFloatTypes())             \
          .InputMemoryType(OrtMemTypeCPU, 1),                         \
      ArgMinMaxOp<>);

REGISTER_ARGMAX_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 1, 10);
REGISTER_ARGMAX_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 11, 12);
REGISTER_ARGMAX_ELEMENTWISE_KERNEL(ArgMax, 13);

REGISTER_ARGMAX_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 1, 10);
REGISTER_ARGMAX_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 11, 12);
REGISTER_ARGMAX_ELEMENTWISE_KERNEL(ArgMin, 13);

}  // namespace js
}  // namespace onnxruntime
