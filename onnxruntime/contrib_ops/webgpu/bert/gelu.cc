// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/bert/gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Gelu);

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

Status GeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  AppendErfFunction(shader.AdditionalImplementation());

  const std::string vecSize = "(uniforms.output_size + 3u) / 4u";  // equivalent to Math.ceil(output_size / 4)

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes(vecSize)
                            << "let a = " << input.GetByOffset("global_idx") << ";\n"
                            << "let value = 0.5 * a * (1.0 + erf_vf32(a * 0.7071067811865475));\n"
                            << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status Gelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  TensorShape input_shape = input->Shape();

  auto* output = context.Output(0, input_shape);
  int64_t output_size = output->Shape().Size();

  GeluProgram program{};
  program.AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
