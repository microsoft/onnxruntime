// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "depth_to_space.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    DepthToSpace,
    kMSInternalNHWCDomain,
    13,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedDataTypes()),
    DepthToSpace<true, true>);

ONNX_OPERATOR_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedDataTypes()),
    DepthToSpace<true, false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kMSInternalNHWCDomain,
    11, 12,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedDataTypes()),
    DepthToSpace<true, true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedDataTypes()),
    DepthToSpace<true, false>);

}  // namespace js
}  // namespace onnxruntime
