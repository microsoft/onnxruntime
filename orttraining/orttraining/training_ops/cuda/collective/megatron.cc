// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nccl_kernels.h"
#include "core/providers/cuda/tensor/identity_op.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    MegatronF,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    MegatronG,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    NcclAllReduce);

}  // namespace cuda
}  // namespace onnxruntime
