// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "contrib_ops/webgpu/bert/fast_gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    FastGelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    FastGelu);

Status FastGeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << TanhImpl;
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                            << "  var a = " << x.GetByOffset("global_idx") << ";\n";
  if (Inputs().size() > 1) {
    const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
    if (bias_components_ == 1) {
      shader.MainFunctionBody() << "  let bias_offset = global_idx * 4;\n"
                                   "  a += x_value_t("
                                << bias.GetByOffset("bias_offset % uniforms.bias_shape") << ", "
                                << bias.GetByOffset("(bias_offset + 1) % uniforms.bias_shape") << ", "
                                << bias.GetByOffset("(bias_offset + 2) % uniforms.bias_shape") << ", "
                                << bias.GetByOffset("(bias_offset + 3) % uniforms.bias_shape") << ");\n";
    } else {
      shader.MainFunctionBody() << "  a += " << bias.GetByOffset("global_idx % uniforms.bias_shape") + ";\n";
    }
  }
  shader.MainFunctionBody() << y.SetByOffset("global_idx", onnxruntime::webgpu::FastGeluExpr);

  return Status::OK();
}

Status FastGelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* bias = context.Input(1);
  auto* output = context.Output(0, input->Shape());

  uint32_t data_size = gsl::narrow<uint32_t>(output->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const auto vec_size = (data_size + 3) / 4;
  uint32_t bias_size = 0;
  int bias_components = 1;

  if (bias != nullptr) {
    bias_size = gsl::narrow<uint32_t>(bias->Shape().Size());
    if (bias_size % 4 == 0) {
      bias_components = 4;
      bias_size = bias_size / 4;
    }
  }

  FastGeluProgram program{bias_components};
  program.AddInput({input, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .AddOutput({output, ProgramTensorMetadataDependency::None, {vec_size}, 4})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariable({vec_size});

  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, {bias_size}, bias_components});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
