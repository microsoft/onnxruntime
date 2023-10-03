// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_add.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(
    BiasAdd,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedFloatTypes()),
    BiasAdd);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
