// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "batch_norm.h"

namespace onnxruntime {
namespace js {

#define REGISTER_BATCHNORM_KERNEL(OP_TYPE, DOMAIN, KERNEL_CLASS)                         \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                     \
      OP_TYPE, DOMAIN, 7, 8, kJsExecutionProvider,                                       \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()), KERNEL_CLASS);  \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                     \
      OP_TYPE, DOMAIN, 9, 13, kJsExecutionProvider,                                      \
      KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()), KERNEL_CLASS);  \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(OP_TYPE, DOMAIN, 14, 14, kJsExecutionProvider,       \
                                    KernelDefBuilder()                                   \
                                        .TypeConstraint("T", JsepSupportedFloatTypes())  \
                                        .TypeConstraint("U", JsepSupportedFloatTypes()), \
                                    KERNEL_CLASS);                                       \
  ONNX_OPERATOR_KERNEL_EX(OP_TYPE, DOMAIN, 15, kJsExecutionProvider,                     \
                          KernelDefBuilder()                                             \
                              .TypeConstraint("T", JsepSupportedFloatTypes())            \
                              .TypeConstraint("T1", JsepSupportedFloatTypes())           \
                              .TypeConstraint("T2", JsepSupportedFloatTypes()),          \
                          KERNEL_CLASS);

REGISTER_BATCHNORM_KERNEL(BatchNormalization, kMSInternalNHWCDomain, BatchNorm<true>);
REGISTER_BATCHNORM_KERNEL(BatchNormalization, kOnnxDomain, BatchNorm<false>);

}  // namespace js
}  // namespace onnxruntime
