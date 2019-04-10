// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/mean_variance_normalization.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
  MeanVarianceNormalization,
  9,
  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
  MeanVarianceNormalization_1<float>);
}
