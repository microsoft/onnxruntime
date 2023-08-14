// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_add.h"

namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    BiasAdd,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasAdd);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
