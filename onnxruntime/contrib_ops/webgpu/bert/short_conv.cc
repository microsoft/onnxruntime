// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/short_conv.h"

#include "contrib_ops/webgpu/bert/kernel_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    ShortConv,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ShortConv);

namespace {
constexpr uint32_t kRmsWorkgroupSize = 64;
}  // namespace

Status ShortConvInvRmsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& inv_rms = shader.AddOutput("inv_rms", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << "var<workgroup> row_partials: array<f32, " << kRmsWorkgroupSize << ">;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let row_base = row * uniforms.hidden_size;\n"
      << "  var sum_sq = 0.0;\n"
      << "  for (var i = local_idx; i < uniforms.hidden_size; i += " << kRmsWorkgroupSize << "u) {\n"
      << "    let v = f32(" << input.GetByOffset("row_base + i") << ");\n"
      << "    sum_sq += v * v;\n"
      << "  }\n"
      << "  row_partials[local_idx] = sum_sq;\n"
      << "  workgroupBarrier();\n"
      << "  for (var stride = " << (kRmsWorkgroupSize / 2) << "u; stride > 0u; stride >>= 1u) {\n"
      << "    if (local_idx < stride) { row_partials[local_idx] += row_partials[local_idx + stride]; }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (local_idx == 0u) {\n"
      << "    " << inv_rms.SetByOffset("row", "inverseSqrt(row_partials[0] / f32(uniforms.hidden_size) + uniforms.epsilon)") << "\n"
      << "  }\n";
  return Status::OK();
}

Status ShortConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& norm_scale = shader.AddInput("norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& inv_rms = shader.AddInput("inv_rms", ShaderUsage::UseUniform);
  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << kernel_helper::kStableSigmoidWgsl << kernel_helper::kSiluWgsl;

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let channels = uniforms.hc_mult * uniforms.hidden_size;\n"
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let t = (global_idx / channels) % uniforms.sequence_length;\n"
      << "  let b = global_idx / (uniforms.sequence_length * channels);\n"
      << "  let flat_channel = g * uniforms.hidden_size + c;\n"
      << "  let scale = f32(" << norm_scale.GetByOffset("flat_channel") << ");\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "  var sum = f32(" << bias->GetByOffset("flat_channel") << ");\n";
  } else {
    shader.MainFunctionBody() << "  var sum = 0.0;\n";
  }
  shader.MainFunctionBody()
      << "  for (var k = 0u; k < uniforms.kernel_size; k++) {\n"
      << "    let offset = (uniforms.kernel_size - 1u - k) * uniforms.dilation;\n"
      << "    if (t >= offset) {\n"
      << "      let source_t = t - offset;\n"
      << "      let source_row = (b * uniforms.sequence_length + source_t) * uniforms.hc_mult + g;\n"
      << "      let normed = f32(" << input.GetByOffset("source_row * uniforms.hidden_size + c") << ") * "
      << inv_rms.GetByOffset("source_row") << " * scale;\n"
      << "      sum += normed * f32(" << weight.GetByOffset("flat_channel * uniforms.kernel_size + k") << ");\n"
      << "    }\n"
      << "  }\n";
  if (apply_silu_or_swish_) {
    shader.MainFunctionBody() << "  sum = silu(sum);\n";
  }
  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_element_t(sum)") << "\n";
  return Status::OK();
}

ShortConv::ShortConv(const OpKernelInfo& info) : WebGpuKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

Status ShortConv::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* weight = context.Input(1);
  const auto* norm_scale = context.Input(2);
  const auto* bias = context.Input(3);
  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  ORT_RETURN_IF_NOT(norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape() == TensorShape({channels}), "bias must have shape (hc_mult * hidden_size)");
  }
  auto* output = context.Output(0, input_shape);
  const int64_t total = input_shape.Size();
  if (total == 0) {
    return Status::OK();
  }

  // First pass: one inverse-RMS value per (batch, sequence, hc_mult) row.
  const int64_t rows = batch_size * sequence_length * hc_mult;
  Tensor inv_rms = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape({rows}));
  ShortConvInvRmsProgram inv_rms_program;
  inv_rms_program.AddInput({input, ProgramTensorMetadataDependency::Type})
      .AddOutput({&inv_rms, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(kRmsWorkgroupSize)
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(rows))
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(rows)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {epsilon_}});
  ORT_RETURN_IF_ERROR(context.RunProgram(inv_rms_program));

  ShortConvProgram program{bias != nullptr, activation_ == "silu" || activation_ == "swish"};
  program.CacheHint(bias != nullptr, activation_)
      .AddInputs({{input, ProgramTensorMetadataDependency::Type},
                  {weight, ProgramTensorMetadataDependency::Type},
                  {norm_scale, ProgramTensorMetadataDependency::Type},
                  {&inv_rms, ProgramTensorMetadataDependency::Type}});
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(weight_shape[2])},
                            {onnxruntime::narrow<uint32_t>(dilation_)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
