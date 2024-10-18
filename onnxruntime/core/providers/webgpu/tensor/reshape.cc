// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/reshape.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    21,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    14, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    13, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    5, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

}  // namespace webgpu
}  // namespace onnxruntime
