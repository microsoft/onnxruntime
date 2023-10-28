// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "einsum.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    12,
    float,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Einsum);

}  // namespace js
}  // namespace onnxruntime
