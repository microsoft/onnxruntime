// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fused_gemm.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    FusedGemm,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedGemm<float, float, float, float>);
}  // namespace contrib
}  // namespace onnxruntime
