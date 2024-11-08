// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grid_sample.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GridSample,
    kMSInternalNHWCDomain,
    16, 19,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", JsepSupportedDataTypes())
        .TypeConstraint("T2", JsepSupportedFloatTypes()),
    GridSample<true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GridSample,
    kOnnxDomain,
    16, 19,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", JsepSupportedDataTypes())
        .TypeConstraint("T2", JsepSupportedFloatTypes()),
    GridSample<false>);

}  // namespace js
}  // namespace onnxruntime
