// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/flatten.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    1, 8,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    9, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    13, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Flatten);

ONNX_OPERATOR_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    21,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Flatten);

}  // namespace webgpu
}  // namespace onnxruntime
