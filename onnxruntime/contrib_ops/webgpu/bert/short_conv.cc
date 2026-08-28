// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/short_conv.h"

#include "contrib_ops/webgpu/bert/engram_helper.h"
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

// Emits a WGSL `raw_at(b, p, g, c)` that reads the raw (un-normalized) value at a virtual position of
// "past_state history followed by the current chunk". Missing history reads as zero, which is exactly
// what a fresh sequence contributes. Requires `sequence_length`, `state_length`, `hc_mult` and
// `hidden_size` uniforms on the program.
void AppendRawAtWgsl(ShaderHelper& shader, const ShaderVariableHelper& input,
                     const ShaderVariableHelper* past_state) {
  shader.AdditionalImplementation()
      << "fn raw_at(b: u32, p: u32, g: u32, c: u32) -> f32 {\n"
      << "  if (p >= uniforms.state_length) {\n"
      << "    let t = p - uniforms.state_length;\n"
      << "    return f32("
      << input.GetByOffset("((b * uniforms.sequence_length + t) * uniforms.hc_mult + g) * uniforms.hidden_size + c")
      << ");\n"
      << "  }\n";
  if (past_state != nullptr) {
    shader.AdditionalImplementation()
        << "  return f32("
        << past_state->GetByOffset(
               "((b * uniforms.state_length + p) * uniforms.hc_mult + g) * uniforms.hidden_size + c")
        << ");\n";
  } else {
    shader.AdditionalImplementation() << "  return 0.0;\n";
  }
  shader.AdditionalImplementation() << "}\n";
}
}  // namespace

Status ShortConvInvRmsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* past_state = nullptr;
  if (has_past_state_) {
    past_state = &shader.AddInput("past_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& inv_rms = shader.AddOutput("inv_rms", ShaderUsage::UseUniform);

  AppendRawAtWgsl(shader, input, past_state);
  shader.AdditionalImplementation() << "var<workgroup> row_partials: array<f32, " << kRmsWorkgroupSize << ">;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let virtual_length = uniforms.state_length + uniforms.sequence_length;\n"
      << "  let g = row % uniforms.hc_mult;\n"
      << "  let p = (row / uniforms.hc_mult) % virtual_length;\n"
      << "  let b = row / (virtual_length * uniforms.hc_mult);\n"
      << "  var sum_sq = 0.0;\n"
      << "  for (var i = local_idx; i < uniforms.hidden_size; i += " << kRmsWorkgroupSize << "u) {\n"
      << "    let v = raw_at(b, p, g, i);\n"
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
  const ShaderVariableHelper* past_state = nullptr;
  if (has_past_state_) {
    past_state = &shader.AddInput("past_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << engram_helper::kStableSigmoidWgsl << engram_helper::kSiluWgsl;
  AppendRawAtWgsl(shader, input, past_state);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let channels = uniforms.hc_mult * uniforms.hidden_size;\n"
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let t = (global_idx / channels) % uniforms.sequence_length;\n"
      << "  let b = global_idx / (uniforms.sequence_length * channels);\n"
      << "  let virtual_length = uniforms.state_length + uniforms.sequence_length;\n"
      << "  let flat_channel = g * uniforms.hidden_size + c;\n"
      << "  let scale = f32(" << norm_scale.GetByOffset("flat_channel") << ");\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "  var sum = f32(" << bias->GetByOffset("flat_channel") << ");\n";
  } else {
    shader.MainFunctionBody() << "  var sum = 0.0;\n";
  }
  shader.MainFunctionBody()
      << "  for (var k = 0u; k < uniforms.kernel_size; k++) {\n"
      // Tap k is `offset` positions back and offset <= state_length, so `p` never underflows.
      << "    let offset = (uniforms.kernel_size - 1u - k) * uniforms.dilation;\n"
      << "    let p = t + uniforms.state_length - offset;\n"
      << "    let virtual_row = (b * virtual_length + p) * uniforms.hc_mult + g;\n"
      << "    let normed = raw_at(b, p, g, c) * " << inv_rms.GetByOffset("virtual_row") << " * scale;\n"
      << "    sum += normed * f32(" << weight.GetByOffset("flat_channel * uniforms.kernel_size + k") << ");\n"
      << "  }\n";
  if (apply_silu_) {
    shader.MainFunctionBody() << "  sum = silu(sum);\n";
  }
  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_element_t(sum)") << "\n";
  return Status::OK();
}

Status ShortConvPresentStateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* past_state = nullptr;
  if (has_past_state_) {
    past_state = &shader.AddInput("past_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& present_state = shader.AddOutput("present_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  AppendRawAtWgsl(shader, input, past_state);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let slot = (global_idx / (uniforms.hc_mult * uniforms.hidden_size)) % uniforms.state_length;\n"
      << "  let b = global_idx / (uniforms.state_length * uniforms.hc_mult * uniforms.hidden_size);\n"
      // present_state holds the trailing state_length virtual positions, still un-normalized.
      << "  let value = raw_at(b, slot + uniforms.sequence_length, g, c);\n"
      << "  " << present_state.SetByOffset("global_idx", "present_state_element_t(value)") << "\n";
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
  const auto* past_state = context.Input(4);
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
  const int64_t kernel_size = weight_shape[2];
  ORT_RETURN_IF_NOT(kernel_size >= 1, "kernel_size must be >= 1");

  // The convolution receptive field reaches this many positions before the current token, so this is
  // exactly the amount of raw input history that has to be carried between invocations.
  const int64_t state_length = (kernel_size - 1) * dilation_;
  const TensorShape state_shape({batch_size, state_length, hc_mult, hidden_size});
  if (past_state != nullptr) {
    ORT_RETURN_IF_NOT(past_state->Shape() == state_shape,
                      "past_state must have shape (batch_size, (kernel_size - 1) * dilation, hc_mult, hidden_size)");
  }
  const bool has_past_state = past_state != nullptr;

  auto* output = context.Output(0, input_shape);
  auto* present_state = context.Output(1, state_shape);
  const int64_t total = input_shape.Size();
  if (total == 0) {
    // No new positions, so present_state is past_state unchanged (zeros when there is no history).
    // It still has to be written: output buffers are not zero-initialized.
    if (present_state != nullptr && state_shape.Size() > 0) {
      if (has_past_state) {
        ORT_RETURN_IF_ERROR(context.CopyTensor(*past_state, *present_state));
      } else {
        context.FillZero(*present_state);
      }
    }
    return Status::OK();
  }

  // First pass: one inverse-RMS value per (batch, virtual position, hc_mult) row. Recomputing it from
  // the raw history (rather than storing normalized history) makes a chunked run bit-exact against a
  // full-sequence run for every element type.
  const int64_t virtual_rows = batch_size * (state_length + sequence_length) * hc_mult;
  Tensor inv_rms = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape({virtual_rows}));
  ShortConvInvRmsProgram inv_rms_program{has_past_state};
  inv_rms_program.CacheHint(has_past_state).AddInput({input, ProgramTensorMetadataDependency::Type});
  if (has_past_state) {
    inv_rms_program.AddInput({past_state, ProgramTensorMetadataDependency::Type});
  }
  inv_rms_program.AddOutput({&inv_rms, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(kRmsWorkgroupSize)
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(virtual_rows))
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(virtual_rows)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(state_length)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {epsilon_}});
  ORT_RETURN_IF_ERROR(context.RunProgram(inv_rms_program));

  ShortConvProgram program{bias != nullptr, has_past_state, activation_ == "silu" || activation_ == "swish"};
  program.CacheHint(bias != nullptr, has_past_state, activation_)
      .AddInputs({{input, ProgramTensorMetadataDependency::Type},
                  {weight, ProgramTensorMetadataDependency::Type},
                  {norm_scale, ProgramTensorMetadataDependency::Type},
                  {&inv_rms, ProgramTensorMetadataDependency::Type}});
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type});
  }
  if (has_past_state) {
    program.AddInput({past_state, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(state_length)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(kernel_size)},
                            {onnxruntime::narrow<uint32_t>(dilation_)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  const int64_t present_total = state_shape.Size();
  if (present_state == nullptr || present_total == 0) {
    return Status::OK();
  }
  ShortConvPresentStateProgram present_program{has_past_state};
  present_program.CacheHint(has_past_state).AddInput({input, ProgramTensorMetadataDependency::Type});
  if (has_past_state) {
    present_program.AddInput({past_state, ProgramTensorMetadataDependency::Type});
  }
  present_program.AddOutput({present_state, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(present_total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(present_total)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(state_length)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)}});
  return context.RunProgram(present_program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
