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
  const auto& input = shader.AddInput("input",
                                      ToProgramVariableDataType(Inputs()[0].tensor->GetElementType(), 4),
                                      ShaderVariable::UseUniform | ShaderVariable::UseValueTypeAlias);

  const auto& bias = Inputs().size() > 1 ? shader.AddInput("bias",
                                                           ToProgramVariableDataType(Inputs()[1].tensor->GetElementType(), bias_components_),
                                                           ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride)
                                         : input;

  const auto& output = shader.AddOutput("output",
                                        ToProgramVariableDataType(Outputs()[0].tensor->GetElementType(), 4),
                                        ShaderVariable::UseUniform);
  const std::string& get_bias = bias_components_ == 1 ? "let x_offset = global_idx * 4;\nlet bias = input_value_t(" + bias.GetByOffset("x_offset % uniforms.bias_shape") + ", " + bias.GetByOffset("(x_offset + 1) % uniforms.bias_shape") + ", " + bias.GetByOffset("(x_offset + 2) % uniforms.bias_shape") + ", " + bias.GetByOffset("(x_offset + 3) % uniforms.bias_shape") + ") " : " let bias = " + bias.GetByOffset(" global_idx % uniforms.bias_shape ");
  const std::string& add_bias = Inputs().size() > 1 ? get_bias + ";\n x += bias;\n" : "";
  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "var x = ", input.GetByOffset("global_idx"), ";\n", add_bias,
                          "let y = x * (0.5 + 0.5 * tanh(x * (0.035677408136300125 * x * x + 0.7978845608028654)));\n",
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
  uint32_t bias_size = nullptr == bias ? 0 : SafeInt<uint32_t>(bias->Shape().Size());
  int bias_components = 1;
  if (bias != nullptr && bias_size % 4 == 0) {
    bias_components = 4;
    bias_size = bias_size / 4;
  }
  FastGeluProgram program{"FastGelu", bias_components};
  if (nullptr == bias) {
    program.Inputs({{input, ProgramTensorMetadataDependency::Type, {vec_size}}});
  } else {
    program.Inputs({{input, ProgramTensorMetadataDependency::Type, {{vec_size}}}, {bias, ProgramTensorMetadataDependency::TypeAndRank, {bias_size}}});
  }
  program
      .Outputs({{output, ProgramTensorMetadataDependency::None, {vec_size}}})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({{vec_size}})
      .CacheHint(std::to_string(bias_components));
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
