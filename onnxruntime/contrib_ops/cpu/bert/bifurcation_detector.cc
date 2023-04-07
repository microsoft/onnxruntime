// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bifurcation_detector.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    BifurcationDetector,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    BifurcationDetector);
}  // namespace contrib
}  // namespace onnxruntime
