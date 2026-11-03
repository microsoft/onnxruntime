// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/hyper_connection_post_mix.h"

#include <string>
#include <string_view>

#include "contrib_ops/cpu/hyper_connection_helper.h"
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

ONNX_OPERATOR_KERNEL_EX(
    HyperConnectionPostMixGateF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Tin0", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin2", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin3", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin4", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tout0", DataTypeImpl::GetTensorType<MLFloat16>()),
    HyperConnectionPostMixGateF16);

ONNX_OPERATOR_KERNEL_EX(
    HyperConnectionSharedPostMixF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Tin0", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin2", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin3", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin4", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin5", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tin6", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tout0", DataTypeImpl::GetTensorType<MLFloat16>()),
    HyperConnectionSharedPostMixF16);

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
  const ShaderVariableHelper* divisor = nullptr;
  const ShaderVariableHelper* scale = nullptr;
  if (fuse_gate_) {
    divisor = &shader.AddInput("divisor", ShaderUsage::UseUniform);
    scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* shared = nullptr;
  const ShaderVariableHelper* shared_gate = nullptr;
  if (fuse_shared_) {
    shared = &shader.AddInput("shared_output", ShaderUsage::UseUniform);
    shared_gate = &shader.AddInput("shared_gate", ShaderUsage::UseUniform);
  }
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
      << "  let raw_gate = " << post_mix.GetByOffset("gate_offset") << ";\n";
  if (fuse_gate_) {
    // Keep the FP16 storage boundaries of the three original elementwise kernels.
    shader.MainFunctionBody()
        << "  let divided_gate = f16(raw_gate / " << divisor->GetByOffset("0u") << ");\n"
        << "  let sigmoid_gate = f16(1.0 / (1.0 + exp(-divided_gate)));\n"
        << "  let gate = f16(sigmoid_gate * " << scale->GetByOffset("0u") << ");\n";
  } else {
    shader.MainFunctionBody() << "  let gate = raw_gate;\n";
  }
  const auto branch_offset = "row * uniforms.hidden + h";
  if (fuse_shared_) {
    // Preserve the stored FP16 sigmoid and GatedAdd results before the FP32 post-mix.
    shader.MainFunctionBody()
        << "  let shared_sigmoid = f16(1.0 / (1.0 + exp(-"
        << shared_gate->GetByOffset("row") << ")));\n"
        << "  let branch_value = f16(" << branch_output.GetByOffset(branch_offset) << " + ("
        << shared->GetByOffset(branch_offset) << " * shared_sigmoid));\n";
  } else {
    shader.MainFunctionBody() << "  let branch_value = " << branch_output.GetByOffset(branch_offset) << ";\n";
  }
  shader.MainFunctionBody()
      << "  value += f32(gate) * f32(branch_value);\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(value)") << "\n";
  return Status::OK();
}

HyperConnectionPostMix::HyperConnectionPostMix(const OpKernelInfo& info)
    : HyperConnectionPostMix(info, false) {}

HyperConnectionPostMix::HyperConnectionPostMix(const OpKernelInfo& info, bool fuse_gate, bool fuse_shared)
    : WebGpuKernel(info), num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      fuse_gate_(fuse_gate), fuse_shared_(fuse_shared) {}

Status HyperConnectionPostMix::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* streams = context.Input(0);
  const auto* branch_output = context.Input(1);
  const auto* post_mix = context.Input(2);
  const auto* stream_mix = fuse_gate_ ? nullptr : context.Input(3);
  const auto* divisor = fuse_gate_ ? context.Input(3) : nullptr;
  const auto* scale = fuse_gate_ ? context.Input(4) : nullptr;
  if (fuse_gate_) {
    ORT_RETURN_IF_NOT(divisor != nullptr && scale != nullptr &&
                          divisor->Shape().Size() == 1 && scale->Shape().Size() == 1,
                      "HyperConnectionPostMixGateF16 requires scalar divisor and scale");
  }
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(branch_output->Shape(), params));
  const auto* shared = fuse_shared_ ? context.Input(5) : nullptr;
  const auto* shared_gate = fuse_shared_ ? context.Input(6) : nullptr;
  if (fuse_shared_) {
    const auto& reduced = branch_output->Shape();
    ORT_RETURN_IF_NOT(shared != nullptr && shared_gate != nullptr && shared->Shape() == reduced &&
                          shared_gate->Shape().NumDimensions() == reduced.NumDimensions(),
                      "HyperConnectionSharedPostMixF16 requires shared output and gate with matching reduced rank");
    const auto last = reduced.NumDimensions() - 1;
    ORT_RETURN_IF_NOT(shared_gate->Shape()[last] == 1,
                      "HyperConnectionSharedPostMixF16 requires shared gate last dimension 1");
    for (size_t axis = 0; axis < last; ++axis) {
      ORT_RETURN_IF_NOT(shared_gate->Shape()[axis] == reduced[axis],
                        "HyperConnectionSharedPostMixF16 requires matching shared gate leading dimensions");
    }
  }
  hyper_connection::GateLayout layout = hyper_connection::GateLayout::Scalar;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(post_mix->Shape(), streams->Shape(), params, false, layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(stream_mix->Shape(), streams->Shape(), params));
  }
  auto* y = context.Output(0, streams->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(streams->Shape().Size());
  if (count == 0) return Status::OK();
  HyperConnectionPostMixProgram program{static_cast<int>(layout), stream_mix != nullptr, fuse_gate_, fuse_shared_};
  program.CacheHint(static_cast<int>(layout), stream_mix != nullptr, fuse_gate_, fuse_shared_)
      .AddInputs({{streams, ProgramTensorMetadataDependency::Type},
                  {branch_output, ProgramTensorMetadataDependency::Type},
                  {post_mix, ProgramTensorMetadataDependency::Type}});
  if (stream_mix != nullptr) program.AddInput({stream_mix, ProgramTensorMetadataDependency::Type});
  if (fuse_gate_) {
    program.AddInput({divisor, ProgramTensorMetadataDependency::Type});
    program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  }
  if (fuse_shared_) {
    program.AddInput({shared, ProgramTensorMetadataDependency::Type});
    program.AddInput({shared_gate, ProgramTensorMetadataDependency::Type});
  }
  const uint32_t workgroup_size = fuse_shared_ ? 64 : 256;
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count},
                            {onnxruntime::narrow<uint32_t>(params.branches)},
                            {onnxruntime::narrow<uint32_t>(params.hidden)}})
      .SetDispatchGroupSize((count + workgroup_size - 1) / workgroup_size)
      .SetWorkgroupSize(workgroup_size);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
