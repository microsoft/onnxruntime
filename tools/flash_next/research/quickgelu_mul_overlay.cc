// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "quickgelu_mul_overlay.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::contrib::webgpu {
ONNX_OPERATOR_KERNEL_EX(
    QuickGeluMulF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Tin0", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tout0", DataTypeImpl::GetTensorType<MLFloat16>()),
    QuickGeluMulF16);

Status QuickGeluMulF16Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& up = shader.AddInput("up", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  let a = " << x.GetByOffset("global_idx") << ";\n"
         "  let v = a * vec4<f16>(f16(1.0));\n"
         "  var sigmoid_value: vec4<f16>;\n"
         "  for (var i = 0u; i < 4u; i++) {\n"
         "    if (v[i] >= f16(0.0)) {\n"
         "      sigmoid_value[i] = f16(1.0) / (f16(1.0) + exp(-v[i]));\n"
         "    } else {\n"
         "      sigmoid_value[i] = f16(1.0) - f16(1.0) / (f16(1.0) + exp(v[i]));\n"
         "    }\n"
         "  }\n"
         // Match the FP16 QuickGelu output storage boundary before the final Mul.
         "  let activated = vec4<f16>(a * sigmoid_value);\n"
      << "  " << y.SetByOffset("global_idx", "activated * " + up.GetByOffset("global_idx")) << "\n";
  return Status::OK();
}

Status QuickGeluMulF16::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* up = context.Input(1);
  ORT_RETURN_IF_NOT(x->Shape() == up->Shape(), "QuickGeluMulF16 requires identical input shapes");
  auto* y = context.Output(0, x->Shape());
  const auto count = onnxruntime::narrow<uint32_t>((x->Shape().Size() + 3) / 4);
  if (count == 0) return Status::OK();
  QuickGeluMulF16Program program;
  program.AddInput({x, ProgramTensorMetadataDependency::Type, {count}, 4})
      .AddInput({up, ProgramTensorMetadataDependency::Type, {count}, 4})
      .AddOutput({y, ProgramTensorMetadataDependency::None, {count}, 4})
      .AddUniformVariables({{count}})
      .SetDispatchGroupSize((count + 63) / 64)
      .SetWorkgroupSize(64);
  return context.RunProgram(program);
}
}  // namespace onnxruntime::contrib::webgpu
