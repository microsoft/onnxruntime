// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/unsqueeze.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Unsqueeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Unsqueeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    1, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Unsqueeze);

}  // namespace webgpu
}  // namespace onnxruntime
