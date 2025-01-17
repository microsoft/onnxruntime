// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fast_gelu.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(
    FastGelu,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedFloatTypes()),
    FastGelu);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
