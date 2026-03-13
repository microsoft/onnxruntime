// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/llm/attention.h"

namespace onnxruntime {
namespace webgpu {

Status Attention::ComputeInternal(ComputeContext& /*context*/) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Attention operator is not yet implemented for WebGPU EP.");
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Attention, kOnnxDomain, 23, 23, kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T1", WebGpuSupportedFloatTypes())
                                      .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
                                  Attention);

ONNX_OPERATOR_KERNEL_EX(Attention, kOnnxDomain, 24, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T1", WebGpuSupportedFloatTypes())
                            .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
                        Attention);

}  // namespace webgpu
}  // namespace onnxruntime
