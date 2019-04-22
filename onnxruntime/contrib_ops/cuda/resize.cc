// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Resize,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).InputMemoryType<OrtMemTypeCPUInput>(1),
    Resize<float>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
