// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "maxpool_with_mask.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    MaxpoolWithMask,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxpoolWithMask);

}  // namespace contrib
}  // namespace onnxruntime
