// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "tile.h"

namespace onnxruntime {
namespace js {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tile,
    kOnnxDomain,
    6,
    12,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<uint32_t>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Tile);

ONNX_OPERATOR_KERNEL_EX(
    Tile,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<uint32_t>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Tile);
}  // namespace js
}  // namespace onnxruntime
