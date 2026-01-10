// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/bert/bias_split_gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    BiasSplitGelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    BiasSplitGelu);

Status BiasSplitGeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& x =
      shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& bias = shader.AddInput("bias", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << ErfImpl;

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "const M_SQRT2: f32 = sqrt(2.0);\n"
                            << "let halfChannels = uniforms.channels / 2u;\n"
                            << "let biasIdx = global_idx % halfChannels;\n"
                            << "let batchIndex = global_idx / halfChannels;\n"
                            << "let inputOffset = biasIdx + batchIndex * halfChannels * 2;\n"
                            << "let valueLeft = " << x.GetByOffset("inputOffset") << " + " << bias.GetByOffset("biasIdx") << ";\n"
                            << "let valueRight = " << x.GetByOffset("inputOffset + halfChannels") << " + " << bias.GetByOffset("biasIdx + halfChannels") << ";\n"
                            << "let geluRight = valueRight * 0.5 * (erf_v(valueRight / M_SQRT2) + 1);\n"
                            << output.SetByOffset("global_idx", "valueLeft * geluRight");

  return Status::OK();
}

Status BiasSplitGelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* bias = context.Input(1);

  TensorShape input_shape = input->Shape();

  if (input_shape.NumDimensions() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BiasSplitGelu input should have 3 dimensions.");
  }

  int64_t channels = input_shape[2];
  input_shape[2] = channels / 2;  // for output shape calculation (N,S,D) -> (N,S,D/2)

  TensorShape bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1 || bias_shape[0] != channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "BiasSplitGelu bias should have 1 dimension with size equal to the number of channels.");
  }

  int components = GetMaxComponents(channels);
  channels /= components;

  auto* output = context.Output(0, input_shape);
  int64_t output_size = output->Shape().Size() / components;

  BiasSplitGeluProgram program{};
  program
      .AddInputs({{input, ProgramTensorMetadataDependency::None, components},
                  {bias, ProgramTensorMetadataDependency::None, components}})
      .AddOutput({output, ProgramTensorMetadataDependency::None, components})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {static_cast<uint32_t>(channels)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime