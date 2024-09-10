// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fast_gelu.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
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
  const auto& input = shader.AddInput("input", ShaderVariable::UseUniform | ShaderVariable::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderVariable::UseUniform);

  std::string add_bias = "";
  if (Inputs().size() > 1) {
    const auto& bias = shader.AddInput("bias", ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride);
    add_bias = bias_components_ == 1 ? "  let bias_offset = global_idx * 4;\n"
                                       "  x += input_value_t(" +
                                           bias.GetByOffset("bias_offset % uniforms.bias_shape") + ", " +
                                           bias.GetByOffset("(bias_offset + 1) % uniforms.bias_shape") + ", " +
                                           bias.GetByOffset("(bias_offset + 2) % uniforms.bias_shape") + ", " +
                                           bias.GetByOffset("(bias_offset + 3) % uniforms.bias_shape") + ");\n"
                                     : "  x += " + bias.GetByOffset("global_idx % uniforms.bias_shape") + ";\n";
  }

  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "  var x = ", input.GetByOffset("global_idx"), ";\n",
                          add_bias,
                          "  let y = x * (0.5 + 0.5 * tanh(x * (0.035677408136300125 * x * x + 0.7978845608028654)));\n  ",
                          output.SetByOffset("global_idx", "y"));

  return Status::OK();
}

Status FastGelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* bias = context.Input(1);
  auto* output = context.Output(0, input->Shape());

  uint32_t data_size = SafeInt<uint32_t>(output->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const auto vec_size = (data_size + 3) / 4;
  uint32_t bias_size = 0;
  int bias_components = 1;

  if (bias != nullptr) {
    bias_size = SafeInt<uint32_t>(bias->Shape().Size());
    if (bias_size % 4 == 0) {
      bias_components = 4;
      bias_size = bias_size / 4;
    }
  }

  FastGeluProgram program{bias_components};
  program.Input({input, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .Output({output, ProgramTensorMetadataDependency::None, {vec_size}, 4})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariable({vec_size});

  if (bias != nullptr) {
    program.Input({bias, ProgramTensorMetadataDependency::TypeAndRank, {bias_size}, bias_components})
        .CacheHint(std::to_string(bias_components));
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
