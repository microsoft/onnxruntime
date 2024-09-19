// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_data_types.h"
#include "core/providers/js/operators/layer_norm.h"

namespace onnxruntime {
namespace contrib {
namespace js {

// LayerNormalization used to be a contrib op
// that (incorrectly) used kOnnxDomain so we need to version it
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    1,
    16,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", onnxruntime::js::JsepSupportedFloatTypes())
        .TypeConstraint("U", onnxruntime::js::JsepSupportedFloatTypes()),
    onnxruntime::js::LayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(
    SimplifiedLayerNormalization,
    kOnnxDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", onnxruntime::js::JsepSupportedFloatTypes())
        .TypeConstraint("U", onnxruntime::js::JsepSupportedFloatTypes()),
    onnxruntime::js::LayerNorm<true>);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
