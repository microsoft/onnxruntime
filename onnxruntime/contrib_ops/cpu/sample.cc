// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sample.h"

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    SampleOp,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    contrib::SampleOp<float>);
}  // namespace contrib
}  // namespace onnxruntime
