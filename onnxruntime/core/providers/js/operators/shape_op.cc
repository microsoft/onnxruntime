// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/js/js_data_types.h"
#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    13, 14,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    15,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace js
}  // namespace onnxruntime
