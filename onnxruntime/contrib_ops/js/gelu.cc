// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gelu.h"

namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gelu);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
