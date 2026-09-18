// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/hyper_connection.h"

#include <string>
#include <string_view>

#include "contrib_ops/hyper_connection_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

using hyper_connection::GateLayout;
using hyper_connection::StreamShape;

#define REGISTER_HC_WEBGPU_KERNEL(Op)                        \
  ONNX_OPERATOR_KERNEL_EX(                                   \
      Op, kMSDomain, 1, kWebGpuExecutionProvider,            \
      (*KernelDefBuilder::Create())                          \
          .TypeConstraint("T", WebGpuSupportedFloatTypes())  \
          .TypeConstraint("M", WebGpuSupportedFloatTypes()), \
      Op);

REGISTER_HC_WEBGPU_KERNEL(BranchwiseRMSNorm)
REGISTER_HC_WEBGPU_KERNEL(ScaledSiLU)
REGISTER_HC_WEBGPU_KERNEL(HyperConnectionPreMix)
REGISTER_HC_WEBGPU_KERNEL(HyperConnectionPostMix)

#undef REGISTER_HC_WEBGPU_KERNEL

namespace {

std::string GateOffset(int layout, std::string_view row,
                       std::string_view branch, std::string_view feature) {
  if (layout == static_cast<int>(GateLayout::Scalar)) {
    return "0u";
  }
  if (layout == static_cast<int>(GateLayout::Branch) ||
      layout == static_cast<int>(GateLayout::BranchSingleton)) {
    return std::string(row) + " * uniforms.branches + " + std::string(branch);
  }
  return "(" + std::string(row) + " * uniforms.branches + " +
         std::string(branch) + ") * uniforms.hidden + " +
         std::string(feature);
}

}  // namespace

Status BranchwiseRMSNormProgram::GenerateShaderCode(
    ShaderHelper& shader) const {
  const auto& x =
      shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* scale = nullptr;
  if (has_scale_) {
    scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  }
  const auto& y =
      shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation()
      << "var<workgroup> sums : array<f32, workgroup_size_x>;\n";
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
    const std::string offset =
        shared_scale_ ? "h" : "(group % uniforms.branches) * uniforms.hidden + h";
    shader.MainFunctionBody() << "    weight = f32(" << scale->GetByOffset(offset)
                              << ");\n";
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
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      x->Shape(), num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }
  auto* y = context.Output(0, x->Shape());
  const uint32_t groups =
      onnxruntime::narrow<uint32_t>(x->Shape().Size() / params.hidden);
  if (groups == 0) return Status::OK();
  const bool shared_scale =
      scale != nullptr && scale->Shape().Size() == params.hidden;
  BranchwiseRMSNormProgram program{scale != nullptr, shared_scale};
  program.CacheHint(scale != nullptr, shared_scale)
      .AddInput({x, ProgramTensorMetadataDependency::Type});
  if (scale != nullptr) {
    program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables(
          {{onnxruntime::narrow<uint32_t>(params.hidden)},
           {onnxruntime::narrow<uint32_t>(params.branches)}, {groups},
           {epsilon_}})
      .SetDispatchGroupSize(groups)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

Status ScaledSiLUProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x =
      shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* scale = nullptr;
  if (has_scale_) {
    scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  }
  const auto& y =
      shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation()
      << "fn stable_sigmoid(v: f32) -> f32 {\n"
         "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
         "  let e = exp(v); return e / (1.0 + e);\n"
         "}\n";
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  var scale_value = uniforms.alpha;\n";
  if (has_scale_) {
    shader.MainFunctionBody() << "  scale_value = f32("
                              << scale->GetByOffset("0u") << ");\n";
  }
  shader.MainFunctionBody()
      << "  let z_t = y_element_t(f32(" << x.GetByOffset("global_idx")
      << ") * scale_value);\n"
         "  let sigmoid_t = y_element_t(stable_sigmoid(f32(z_t)));\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(f32(z_t) * f32(sigmoid_t))")
      << "\n";
  return Status::OK();
}

ScaledSiLU::ScaledSiLU(const OpKernelInfo& info)
    : WebGpuKernel(info),
      alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

Status ScaledSiLU::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0,
                    "scale must be a scalar");
  auto* y = context.Output(0, x->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(x->Shape().Size());
  if (count == 0) return Status::OK();
  ScaledSiLUProgram program{scale != nullptr};
  program.CacheHint(scale != nullptr)
      .AddInput({x, ProgramTensorMetadataDependency::Type});
  if (scale != nullptr) {
    program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count}, {alpha_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

Status HyperConnectionPreMixProgram::GenerateShaderCode(
    ShaderHelper& shader) const {
  const auto& streams = shader.AddInput("streams", ShaderUsage::UseUniform);
  const auto& pre_mix = shader.AddInput("pre_mix", ShaderUsage::UseUniform);
  const auto& y =
      shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  let row = global_idx / uniforms.hidden;\n"
         "  let h = global_idx % uniforms.hidden;\n"
         "  var sum = 0.0;\n"
         "  for (var c = 0u; c < uniforms.branches; c++) {\n"
         "    let x_offset = (row * uniforms.branches + c) * uniforms.hidden + h;\n"
      << "    let gate_offset = " << GateOffset(gate_layout_, "row", "c", "h")
      << ";\n"
      << "    sum += f32(" << streams.GetByOffset("x_offset") << ") * f32("
      << pre_mix.GetByOffset("gate_offset") << ");\n"
                                               "  }\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(sum * uniforms.reduction_scale)")
      << "\n";
  return Status::OK();
}

HyperConnectionPreMix::HyperConnectionPreMix(const OpKernelInfo& info)
    : WebGpuKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(
          info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

Status HyperConnectionPreMix::ComputeInternal(ComputeContext& context) const {
  const auto* streams = context.Input(0);
  const auto* pre_mix = context.Input(1);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  GateLayout layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      pre_mix->Shape(), streams->Shape(), params, false, layout));
  auto* y = context.Output(0, TensorShape(params.reduced_shape));
  const uint32_t count = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (count == 0) return Status::OK();
  HyperConnectionPreMixProgram program{static_cast<int>(layout)};
  program.CacheHint(static_cast<int>(layout))
      .AddInputs({{streams, ProgramTensorMetadataDependency::Type},
                  {pre_mix, ProgramTensorMetadataDependency::Type}})
      .AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables(
          {{count}, {onnxruntime::narrow<uint32_t>(params.branches)}, {onnxruntime::narrow<uint32_t>(params.hidden)}, {reduction_scale_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

Status HyperConnectionPostMixProgram::GenerateShaderCode(
    ShaderHelper& shader) const {
  const auto& streams = shader.AddInput("streams", ShaderUsage::UseUniform);
  const auto& branch_output =
      shader.AddInput("branch_output", ShaderUsage::UseUniform);
  const auto& post_mix = shader.AddInput("post_mix", ShaderUsage::UseUniform);
  const ShaderVariableHelper* stream_mix = nullptr;
  if (has_stream_mix_) {
    stream_mix = &shader.AddInput("stream_mix", ShaderUsage::UseUniform);
  }
  const auto& y =
      shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  let h = global_idx % uniforms.hidden;\n"
         "  let c = (global_idx / uniforms.hidden) % uniforms.branches;\n"
         "  let row = global_idx / (uniforms.hidden * uniforms.branches);\n";
  if (has_stream_mix_) {
    shader.MainFunctionBody()
        << "  var value = 0.0;\n"
           "  for (var source = 0u; source < uniforms.branches; source++) {\n"
           "    let matrix_offset = (row * uniforms.branches + source) * uniforms.branches + c;\n"
           "    let stream_offset = (row * uniforms.branches + source) * uniforms.hidden + h;\n"
        << "    value += f32(" << stream_mix->GetByOffset("matrix_offset")
        << ") * f32(" << streams.GetByOffset("stream_offset") << ");\n"
        << "  }\n";
  } else {
    shader.MainFunctionBody()
        << "  var value = f32(" << streams.GetByOffset("global_idx") << ");\n";
  }
  shader.MainFunctionBody()
      << "  let gate_offset = " << GateOffset(gate_layout_, "row", "c", "h")
      << ";\n"
      << "  value += f32(" << post_mix.GetByOffset("gate_offset") << ") * f32("
      << branch_output.GetByOffset("row * uniforms.hidden + h") << ");\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(value)") << "\n";
  return Status::OK();
}

HyperConnectionPostMix::HyperConnectionPostMix(const OpKernelInfo& info)
    : WebGpuKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

Status HyperConnectionPostMix::ComputeInternal(ComputeContext& context) const {
  const auto* streams = context.Input(0);
  const auto* branch_output = context.Input(1);
  const auto* post_mix = context.Input(2);
  const auto* stream_mix = context.Input(3);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(
      hyper_connection::ValidateReduced(branch_output->Shape(), params));
  GateLayout layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      post_mix->Shape(), streams->Shape(), params, false, layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(
        stream_mix->Shape(), streams->Shape(), params));
  }
  auto* y = context.Output(0, streams->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(streams->Shape().Size());
  if (count == 0) return Status::OK();
  HyperConnectionPostMixProgram program{static_cast<int>(layout),
                                        stream_mix != nullptr};
  program.CacheHint(static_cast<int>(layout), stream_mix != nullptr)
      .AddInputs({{streams, ProgramTensorMetadataDependency::Type},
                  {branch_output, ProgramTensorMetadataDependency::Type},
                  {post_mix, ProgramTensorMetadataDependency::Type}});
  if (stream_mix != nullptr) {
    program.AddInput({stream_mix, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables(
          {{count}, {onnxruntime::narrow<uint32_t>(params.branches)}, {onnxruntime::narrow<uint32_t>(params.hidden)}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
