// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/branchwise_rms_norm.h"

#include "contrib_ops/hyper_connection_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(
    BranchwiseRMSNorm, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    BranchwiseRMSNorm);

Status BranchwiseRMSNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* scale = nullptr;
  if (has_scale_) {
    scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  }
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << "var<workgroup> sums : array<f32, workgroup_size_x>;\n";
  shader.MainFunctionBody()
      << "  let group = workgroup_idx;\n"
         "  if (group >= uniforms.groups) { return; }\n"
         "  let base = group * uniforms.hidden;\n"
         "  var sum = 0.0;\n"
         "  for (var h = local_idx; h < uniforms.hidden; h += workgroup_size_x) {\n"
      << "    let v = f32(" << x.GetByOffset("base + h") << ");\n"
      << "    sum += v * v;\n"
         "  }\n"
         "  sums[local_idx] = sum;\n"
         "  workgroupBarrier();\n"
         "  var width = workgroup_size_x;\n"
         "  for (var half = width >> 1u; half > 0u; half = width >> 1u) {\n"
         "    width = half + (width & 1u);\n"
         "    if (local_idx < half) { sums[local_idx] += sums[local_idx + width]; }\n"
         "    workgroupBarrier();\n"
         "  }\n"
         "  let inv_rms = inverseSqrt(sums[0] / f32(uniforms.hidden) + uniforms.epsilon);\n"
         "  for (var h = local_idx; h < uniforms.hidden; h += workgroup_size_x) {\n"
         "    var weight = 1.0;\n";
  if (has_scale_) {
    const char* scale_offset =
        shared_scale_ ? "h" : "(group % uniforms.branches) * uniforms.hidden + h";
    shader.MainFunctionBody()
        << "    weight = f32(" << scale->GetByOffset(scale_offset) << ");\n";
  }
  shader.MainFunctionBody()
      << "    " << y.SetByOffset("base + h", "y_element_t(f32(" + x.GetByOffset("base + h") + ") * inv_rms * weight)")
      << "\n  }\n";
  return Status::OK();
}

BranchwiseRMSNorm::BranchwiseRMSNorm(const OpKernelInfo& info)
    : WebGpuKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

Status BranchwiseRMSNorm::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(x->Shape(), num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }
  auto* y = context.Output(0, x->Shape());
  const uint32_t groups = onnxruntime::narrow<uint32_t>(x->Shape().Size() / params.hidden);
  if (groups == 0) return Status::OK();

  const bool shared_scale = scale != nullptr && scale->Shape().Size() == params.hidden;
  if (shared_scale && scale->GetElementType() == x->GetElementType()) {
    return onnxruntime::webgpu::RunLayerNormProgram(
        context, x, scale, nullptr, epsilon_, groups, params.hidden, true, y, nullptr, nullptr);
  }

  BranchwiseRMSNormProgram program{scale != nullptr, shared_scale};
  program.CacheHint(scale != nullptr, shared_scale)
      .AddInput({x, ProgramTensorMetadataDependency::Type});
  if (scale != nullptr) {
    program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(params.hidden)},
                            {onnxruntime::narrow<uint32_t>(params.branches)},
                            {groups},
                            {epsilon_}})
      .SetDispatchGroupSize(groups)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
