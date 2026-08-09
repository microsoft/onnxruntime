// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/identity.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    13, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    14, 15,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    16, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    23, 23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    24, 24,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

ONNX_OPERATOR_KERNEL_EX(
    Identity,
    kOnnxDomain,
    25,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", WebGpuSupportedNumberTypes())
        .Alias(0, 0),
    Identity);

}  // namespace webgpu
}  // namespace onnxruntime
