// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/js/skip_layer_norm.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(
    SkipLayerNormalization,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedFloatTypes())
        .TypeConstraint("U", JsepSupportedFloatTypes()),
    SkipLayerNorm);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
