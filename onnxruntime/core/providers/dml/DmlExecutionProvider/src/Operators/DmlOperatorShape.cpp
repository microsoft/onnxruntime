// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/shape_op.h"

namespace Dml
{

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1, 12,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    13, 14,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    15,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

} // namespace Dml
