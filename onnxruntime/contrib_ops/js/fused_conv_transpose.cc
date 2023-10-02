// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fused_conv_transpose.h"
namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    FusedConvTranspose,
    kMSDomain,
    1,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConvTranspose<false>);

ONNX_OPERATOR_KERNEL_EX(
    FusedConvTranspose,
    kMSInternalNHWCDomain,
    1,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConvTranspose<true>);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
