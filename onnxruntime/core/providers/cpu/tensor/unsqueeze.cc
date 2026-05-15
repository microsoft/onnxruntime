// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/unsqueeze.h"
#include "utils.h"
#include "core/providers/common.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    1,
    10,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    11,
    12,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

// axes is input instead of attribute
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    13,
    20,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

// Opset 21 added support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
// TODO(adrianlizarraga): Implement support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    21,
    22,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    23,
    23,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    24,
    24,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

// Opset 25
ONNX_CPU_OPERATOR_KERNEL(
    Unsqueeze,
    25,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

Status Unsqueeze::Compute(OpKernelContext* ctx) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));
  CopyCpuTensor(p.input_tensor, p.output_tensor);
  return Status::OK();
}
}  // namespace onnxruntime
