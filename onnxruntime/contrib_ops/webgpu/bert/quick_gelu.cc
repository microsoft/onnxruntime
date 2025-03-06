// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "contrib_ops/webgpu/bert/quick_gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    QuickGelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    QuickGelu);

Status QuickGeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << QuickGeluImpl;
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                            << "  var a = " << x.GetByOffset("global_idx") << ";\n"
                            << y.SetByOffset("global_idx", onnxruntime::webgpu::QuickGeluExpr);

  return Status::OK();
}

Status QuickGelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  auto* output = context.Output(0, input->Shape());

  uint32_t data_size = gsl::narrow<uint32_t>(output->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const auto vec_size = (data_size + 3) / 4;

  QuickGeluProgram program{};
  program.AddInput({input, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .AddOutput({output, ProgramTensorMetadataDependency::None, {vec_size}, 4})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{vec_size}, {alpha_}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
