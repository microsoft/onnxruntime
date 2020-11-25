// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "orttraining/training_ops/cpu/controlflow/common.h"
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

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(onnxruntime::contrib::AliasRange<0, 0>(0, onnxruntime::contrib::passthrough_input_count_limit)),  // outputs and inputs are mapped one to one
    onnxruntime::contrib::PassThrough);

}  // namespace cuda
}  // namespace onnxruntime
