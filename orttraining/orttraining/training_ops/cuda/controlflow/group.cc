// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Group,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    onnxruntime::contrib::Group);

}  // namespace cuda
}  // namespace onnxruntime
