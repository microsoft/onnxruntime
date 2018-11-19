// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/image_scaler.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    ImageScaler,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ImageScaler<float>);
}  // namespace onnxruntime
