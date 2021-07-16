// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/conv_transpose_with_dynamic_pads.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
ONNX_OPERATOR_TYPED_KERNEL_EX(
    ConvTransposeWithDynamicPads,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    ConvTransposeWithDynamicPads<float>);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
