// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_scaler.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_KERNEL(
    ImageScaler,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ImageScaler<float>);
}
}  // namespace onnxruntime
