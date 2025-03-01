// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/bert/bias_split_gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

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

void AppendErfFunction(std::ostream& os) {
  os << "const r0: input_indices_t = 0.3275911;\n"
     << "const r1: input_indices_t = 0.254829592;\n"
     << "const r2: input_indices_t = -0.284496736;\n"
     << "const r3: input_indices_t = 1.421413741;\n"
     << "const r4: input_indices_t = -1.453152027;\n"
     << "const r5: input_indices_t = 1.061405429;\n\n"
     << "fn erf_vf32(v: vec4<input_indices_t>) -> vec4<input_indices_t> {\n"
     << "  let absv = abs(v);\n"
     << "  let x = 1.0 / (1.0 + r0 * absv);\n"
     << "  return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));\n"
     << "}\n";
}

Status BiasSplitGeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& bias = shader.AddInput("bias");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  AppendErfFunction(shader.AdditionalImplementation());

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "const M_SQRT2 = sqrt(2.0);\n"
                            << "const halfChannels = (" << channels_ << " / 4 / 2)u;\n"
                            << "let biasIdx = global_idx % halfChannels;\n"
                            << "let batchIndex = global_idx / halfChannels;\n"
                            << "let inputOffset = biasIdx + batchIndex * halfChannels * 2;\n"
                            << "let valueLeft = " << input.GetByOffset("inputOffset") << " + " << bias.GetByOffset("biasIdx") << ";\n"
                            << "let valueRight = " << input.GetByOffset("inputOffset + halfChannels") << " + " << bias.GetByOffset("biasIdx + halfChannels") << ";\n"
                            << "let geluRight = valueRight * 0.5 * (erf_vf32(valueRight / M_SQRT2) + 1);\n"
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
  if (channels != 2560 && channels != 5120 && channels != 10240) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BiasSplitGelu hidden state should be 2560, 5120 or 10240.");
  }

  TensorShape bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1 || bias_shape[0] != channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BiasSplitGelu bias should have 1 dimension with size equal to the number of channels.");
  }

  auto* output = context.Output(0, input_shape);
  int64_t output_size = output->Shape().Size();

  BiasSplitGeluProgram program{channels};
  program.AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                     {bias}})
      .AddOutput({output})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
