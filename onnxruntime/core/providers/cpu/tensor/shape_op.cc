// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_KERNEL(
    Shape,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace onnxruntime
