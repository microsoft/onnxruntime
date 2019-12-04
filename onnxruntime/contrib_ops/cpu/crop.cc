// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "crop.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_KERNEL(
    Crop,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Crop<float>);
}
}  // namespace onnxruntime
