// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/moe_base.h"
#include "contrib_ops/webgpu/moe/moe.h"
#include "contrib_ops/cpu/moe/moe_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

Status MoEProgram::GenerateShaderCode(ShaderHelper& /*unused*/) const {
  return Status::OK();
}

Status MoE::ComputeInternal(ComputeContext& /*unused*/) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "MoE is not implemented in WebGPU");
}

ONNX_OPERATOR_KERNEL_EX(
    MoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
