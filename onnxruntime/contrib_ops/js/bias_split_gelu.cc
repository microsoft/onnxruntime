// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_split_gelu.h"

namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    BiasSplitGelu,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasSplitGelu);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
