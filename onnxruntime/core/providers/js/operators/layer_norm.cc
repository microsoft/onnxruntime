// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "layer_norm.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    17,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedFloatTypes())
        .TypeConstraint("U", JsepSupportedFloatTypes()),
    LayerNorm);

}  // namespace js
}  // namespace onnxruntime
