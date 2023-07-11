// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/mean_variance_normalization.h"

namespace onnxruntime {
namespace contrib {
// Register MVN operator for backward compatibility.
// The experimental MVN op was removed. The history has to be kept locally as below.
// As of (9/26/2018) MVN is a production function in ONNX.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MeanVarianceNormalization,
    1,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization);
}  // namespace contrib
}  // namespace onnxruntime
