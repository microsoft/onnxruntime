// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"
#include "core/providers/js/js_data_types.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(
    Attention,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedFloatTypes()),
    Attention);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
