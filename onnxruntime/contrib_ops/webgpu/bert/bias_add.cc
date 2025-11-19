// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/bert/bias_add.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    BiasAdd,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    BiasAdd);

Status BiasAddProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& bias = shader.AddInput("bias", ShaderUsage::UseUniform);
  const ShaderVariableHelper& residual = shader.AddInput("residual", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let value = " << input.GetByOffset("global_idx")
                            << "  + " << bias.GetByOffset("global_idx % uniforms.channels")
                            << "  + " << residual.GetByOffset("global_idx") << ";\n"
                            << "  " + output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status BiasAdd::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* bias = context.Input(1);
  const auto* residual = context.Input(2);

  TensorShape input_shape = input->Shape();

  if (input_shape.NumDimensions() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BiasAdd input should have 3 dimensions.");
  }

  int64_t channels = input_shape[2];
  TensorShape bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1 || bias_shape[0] != channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "BiasAdd bias should have 1 dimension with size equal to the number of channels.");
  }

  int components = GetMaxComponents(channels);
  channels /= components;

  auto* output = context.Output(0, input_shape);
  int64_t output_size = output->Shape().Size() / components;

  BiasAddProgram program{};
  program
      .AddInputs({{input, ProgramTensorMetadataDependency::None, components},
                  {bias, ProgramTensorMetadataDependency::None, components},
                  {residual, ProgramTensorMetadataDependency::None, components}})
      .AddOutput({output, ProgramTensorMetadataDependency::None, components})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {static_cast<uint32_t>(channels)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
