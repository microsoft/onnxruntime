// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/hyper_connection_post_mix.h"

#include <string>
#include <string_view>

#include "contrib_ops/hyper_connection_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(
    HyperConnectionPostMix, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    HyperConnectionPostMix);

namespace {

std::string GateOffset(int layout, std::string_view row,
                       std::string_view branch, std::string_view feature) {
  if (layout == static_cast<int>(hyper_connection::GateLayout::Scalar)) return "0u";
  if (layout == static_cast<int>(hyper_connection::GateLayout::Branch) ||
      layout == static_cast<int>(hyper_connection::GateLayout::BranchSingleton)) {
    return std::string(row) + " * uniforms.branches + " + std::string(branch);
  }
  return "(" + std::string(row) + " * uniforms.branches + " + std::string(branch) +
         ") * uniforms.hidden + " + std::string(feature);
}

}  // namespace

Status HyperConnectionPostMixProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& streams = shader.AddInput("streams", ShaderUsage::UseUniform);
  const auto& branch_output = shader.AddInput("branch_output", ShaderUsage::UseUniform);
  const auto& post_mix = shader.AddInput("post_mix", ShaderUsage::UseUniform);
  const ShaderVariableHelper* stream_mix = nullptr;
  if (has_stream_mix_) stream_mix = &shader.AddInput("stream_mix", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
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
        << "    value += f32(" << stream_mix->GetByOffset("matrix_offset") << ") * f32("
        << streams.GetByOffset("stream_offset") << ");\n"
                                                   "  }\n";
  } else {
    shader.MainFunctionBody() << "  var value = f32(" << streams.GetByOffset("global_idx") << ");\n";
  }
  shader.MainFunctionBody()
      << "  let gate_offset = " << GateOffset(gate_layout_, "row", "c", "h") << ";\n"
      << "  value += f32(" << post_mix.GetByOffset("gate_offset") << ") * f32("
      << branch_output.GetByOffset("row * uniforms.hidden + h") << ");\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(value)") << "\n";
  return Status::OK();
}

HyperConnectionPostMix::HyperConnectionPostMix(const OpKernelInfo& info)
    : WebGpuKernel(info), num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

Status HyperConnectionPostMix::ComputeInternal(ComputeContext& context) const {
  const auto* streams = context.Input(0);
  const auto* branch_output = context.Input(1);
  const auto* post_mix = context.Input(2);
  const auto* stream_mix = context.Input(3);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(branch_output->Shape(), params));
  hyper_connection::GateLayout layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(post_mix->Shape(), streams->Shape(), params, false, layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(stream_mix->Shape(), streams->Shape(), params));
  }
  auto* y = context.Output(0, streams->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(streams->Shape().Size());
  if (count == 0) return Status::OK();
  HyperConnectionPostMixProgram program{static_cast<int>(layout), stream_mix != nullptr};
  program.CacheHint(static_cast<int>(layout), stream_mix != nullptr)
      .AddInputs({{streams, ProgramTensorMetadataDependency::Type},
                  {branch_output, ProgramTensorMetadataDependency::Type},
                  {post_mix, ProgramTensorMetadataDependency::Type}});
  if (stream_mix != nullptr) program.AddInput({stream_mix, ProgramTensorMetadataDependency::Type});
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count},
                            {onnxruntime::narrow<uint32_t>(params.branches)},
                            {onnxruntime::narrow<uint32_t>(params.hidden)}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
