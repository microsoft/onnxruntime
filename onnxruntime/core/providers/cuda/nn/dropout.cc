// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    12,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType<OrtMemTypeCPUInput>(1)
        .InputMemoryType<OrtMemTypeCPUInput>(2),
    Dropout<false>);

}  // namespace cuda
}  // namespace onnxruntime
