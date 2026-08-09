// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

JSEP_KERNEL_IMPL(MatMul, MatMul)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MatMul, kOnnxDomain, 1, 12, kJsExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()),
                                  MatMul);

ONNX_OPERATOR_KERNEL_EX(MatMul, kOnnxDomain, 13, kJsExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", JsepSupportedFloatTypes()),
                        MatMul);

}  // namespace js
}  // namespace onnxruntime
