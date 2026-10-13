// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/scaled_silu.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(
    ScaledSiLU, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    ScaledSiLU);

Status ScaledSiLUProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* scale = nullptr;
  if (has_scale_) scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation()
      << "fn stable_sigmoid(v: f32) -> f32 {\n"
         "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
         "  let e = exp(v); return e / (1.0 + e);\n"
         "}\n";
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  var scale_value = uniforms.alpha;\n";
  if (has_scale_) {
    shader.MainFunctionBody() << "  scale_value = f32(" << scale->GetByOffset("0u") << ");\n";
  }
  shader.MainFunctionBody()
      << "  let z_t = y_element_t(f32(" << x.GetByOffset("global_idx")
      << ") * scale_value);\n"
         "  let sigmoid_t = y_element_t(stable_sigmoid(f32(z_t)));\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(f32(z_t) * f32(sigmoid_t))") << "\n";
  return Status::OK();
}

ScaledSiLU::ScaledSiLU(const OpKernelInfo& info)
    : WebGpuKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

Status ScaledSiLU::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0, "scale must be a scalar");
  auto* y = context.Output(0, x->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(x->Shape().Size());
  if (count == 0) return Status::OK();
  ScaledSiLUProgram program{scale != nullptr};
  program.CacheHint(scale != nullptr).AddInput({x, ProgramTensorMetadataDependency::Type});
  if (scale != nullptr) program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count}, {alpha_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
