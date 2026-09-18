// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/hyper_connection_pre_mix.h"

#include <string>
#include <string_view>

#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(
    HyperConnectionPreMix, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    HyperConnectionPreMix);

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

Status HyperConnectionPreMixProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& streams = shader.AddInput("streams", ShaderUsage::UseUniform);
  const auto& pre_mix = shader.AddInput("pre_mix", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  let row = global_idx / uniforms.hidden;\n"
         "  let h = global_idx % uniforms.hidden;\n"
         "  var sum = 0.0;\n"
         "  for (var c = 0u; c < uniforms.branches; c++) {\n"
         "    let x_offset = (row * uniforms.branches + c) * uniforms.hidden + h;\n"
      << "    let gate_offset = " << GateOffset(gate_layout_, "row", "c", "h") << ";\n"
      << "    sum += f32(" << streams.GetByOffset("x_offset") << ") * f32("
      << pre_mix.GetByOffset("gate_offset") << ");\n"
                                               "  }\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(sum * uniforms.reduction_scale)") << "\n";
  return Status::OK();
}

HyperConnectionPreMix::HyperConnectionPreMix(const OpKernelInfo& info)
    : WebGpuKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

Status HyperConnectionPreMix::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* streams = context.Input(0);
  const auto* pre_mix = context.Input(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  hyper_connection::GateLayout layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(pre_mix->Shape(), streams->Shape(), params, false, layout, true));
  auto* y = context.Output(0, TensorShape(params.reduced_shape));
  const uint32_t count = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (count == 0) return Status::OK();
  HyperConnectionPreMixProgram program{static_cast<int>(layout)};
  program.CacheHint(static_cast<int>(layout))
      .AddInputs({{streams, ProgramTensorMetadataDependency::Type},
                  {pre_mix, ProgramTensorMetadataDependency::Type}})
      .AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count},
                            {onnxruntime::narrow<uint32_t>(params.branches)},
                            {onnxruntime::narrow<uint32_t>(params.hidden)},
                            {reduction_scale_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
