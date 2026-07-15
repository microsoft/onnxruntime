// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/resize.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    10, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", WebGpuSupportedNumberTypes()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    13, 17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", WebGpuSupportedNumberTypes()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    18, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", WebGpuSupportedNumberTypes()),
    Resize);

ONNX_OPERATOR_KERNEL_EX(
    Resize,
    kOnnxDomain,
    19,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", WebGpuSupportedNumberTypes()),
    Resize);

}  // namespace webgpu
}  // namespace onnxruntime
