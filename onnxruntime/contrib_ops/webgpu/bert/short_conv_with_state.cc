// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/short_conv_with_state.h"

#include "contrib_ops/webgpu/bert/kernel_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    ShortConvWithState,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ShortConvWithState);

namespace {
constexpr uint32_t kRmsWorkgroupSize = 64;
}  // namespace

// Pass 1: Compute inverse RMS per row and write normalized values.
Status ShortConvWithStateNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& norm_scale = shader.AddInput("norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& normed = shader.AddOutput("normed", ShaderUsage::UseUniform);

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
      // Write normalized values.
      << "  let inv_rms = inverseSqrt(row_partials[0] / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "  let g = row % uniforms.hc_mult;\n"
      << "  let out_base = (row / uniforms.hc_mult) * uniforms.channels + g * uniforms.hidden_size;\n"
      << "  for (var i = local_idx; i < uniforms.hidden_size; i += " << kRmsWorkgroupSize << "u) {\n"
      << "    let scale = f32(" << norm_scale.GetByOffset("g * uniforms.hidden_size + i") << ");\n"
      << "    let val = f32(" << input.GetByOffset("row_base + i") << ") * inv_rms * scale;\n"
      << "    " << normed.SetByOffset("out_base + i", "val") << "\n"
      << "  }\n";
  return Status::OK();
}

// Pass 2: Convolution kernel.
Status ShortConvWithStateConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& normed = shader.AddInput("normed", ShaderUsage::UseUniform);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  int input_idx = 2;
  const ShaderVariableHelper* past_state_var = nullptr;
  if (has_past_state_) {
    past_state_var = &shader.AddInput("past_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
    input_idx++;
  }
  const ShaderVariableHelper* bias_var = nullptr;
  if (has_bias_) {
    bias_var = &shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
    input_idx++;
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << kernel_helper::kStableSigmoidWgsl << kernel_helper::kSiluWgsl;

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let t = (global_idx / uniforms.channels) % uniforms.sequence_length;\n"
      << "  let b = global_idx / (uniforms.sequence_length * uniforms.channels);\n"
      << "  let flat_channel = g * uniforms.hidden_size + c;\n";

  if (has_bias_) {
    shader.MainFunctionBody() << "  var sum = f32(" << bias_var->GetByOffset("flat_channel") << ");\n";
  } else {
    shader.MainFunctionBody() << "  var sum = 0.0;\n";
  }

  shader.MainFunctionBody()
      << "  for (var k = 0u; k < uniforms.kernel_size; k++) {\n"
      << "    let offset = (uniforms.kernel_size - 1u - k) * uniforms.dilation;\n"
      << "    let src = i32(uniforms.state_len + t) - i32(offset);\n"
      << "    var src_val = 0.0;\n"
      << "    if (src >= 0 && u32(src) < uniforms.state_len) {\n";
  if (has_past_state_) {
    shader.MainFunctionBody()
        << "      src_val = f32(" << past_state_var->GetByOffset("(b * uniforms.channels + flat_channel) * uniforms.state_len + u32(src)") << ");\n";
  }
  shader.MainFunctionBody()
      << "    } else if (u32(src) >= uniforms.state_len && u32(src) < uniforms.state_len + uniforms.sequence_length) {\n"
      << "      let normed_t = u32(src) - uniforms.state_len;\n"
      << "      src_val = " << normed.GetByOffset("(b * uniforms.sequence_length + normed_t) * uniforms.channels + flat_channel") << ";\n"
      << "    }\n"
      << "    sum += src_val * f32(" << weight.GetByOffset("flat_channel * uniforms.kernel_size + k") << ");\n"
      << "  }\n";

  if (apply_silu_) {
    shader.MainFunctionBody() << "  sum = silu(sum);\n";
  }
  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_element_t(sum)") << "\n";
  return Status::OK();
}

// Pass 3: Update present state.
Status ShortConvWithStateUpdateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& normed = shader.AddInput("normed", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_state_var = nullptr;
  if (has_past_state_) {
    past_state_var = &shader.AddInput("past_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& present_state = shader.AddOutput("present_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total_state_elements")
      << "  let s = global_idx % uniforms.state_len;\n"
      << "  let flat_c = (global_idx / uniforms.state_len) % uniforms.channels;\n"
      << "  let b = global_idx / (uniforms.channels * uniforms.state_len);\n"
      << "  let timeline_pos = uniforms.sequence_length + s;\n"
      << "  var val = 0.0;\n"
      << "  if (timeline_pos < uniforms.state_len) {\n";
  if (has_past_state_) {
    shader.MainFunctionBody()
        << "    val = f32(" << past_state_var->GetByOffset("(b * uniforms.channels + flat_c) * uniforms.state_len + timeline_pos") << ");\n";
  }
  shader.MainFunctionBody()
      << "  } else {\n"
      << "    let normed_t = timeline_pos - uniforms.state_len;\n"
      << "    if (normed_t < uniforms.sequence_length) {\n"
      << "      val = " << normed.GetByOffset("(b * uniforms.sequence_length + normed_t) * uniforms.channels + flat_c") << ";\n"
      << "    }\n"
      << "  }\n"
      << "  " << present_state.SetByOffset("(b * uniforms.channels + flat_c) * uniforms.state_len + s", "output_element_t(val)") << "\n";
  return Status::OK();
}

ShortConvWithState::ShortConvWithState(const OpKernelInfo& info) : WebGpuKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

Status ShortConvWithState::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* past_state = context.Input(1);
  const auto* norm_scale = context.Input(2);
  const auto* weight = context.Input(3);
  const auto* bias = context.Input(4);

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
  const int64_t kernel_size = weight_shape[2];
  const int64_t state_len = dilation_ * (kernel_size - 1);

  ORT_RETURN_IF_NOT(norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1 && weight_shape[2] == kernel_size,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (past_state != nullptr) {
    ORT_RETURN_IF_NOT(past_state->Shape() == TensorShape({batch_size, channels, state_len}),
                      "past_state must have shape (batch_size, channels, dilation*(kernel_size-1))");
  }
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape() == TensorShape({channels}), "bias must have shape (hc_mult * hidden_size)");
  }

  auto* output = context.Output(0, input_shape);
  TensorShape present_shape({batch_size, channels, state_len});
  auto* present_state_out = context.Output(1, present_shape);

  const int64_t total = input_shape.Size();
  const int64_t rows = batch_size * sequence_length * hc_mult;

  // Normed buffer: [B, S, C] in float32.
  Tensor normed = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape({batch_size * sequence_length * channels}));

  // Pass 1: Normalize.
  if (rows > 0) {
    ShortConvWithStateNormProgram norm_program;
    norm_program.AddInput({input, ProgramTensorMetadataDependency::Type})
        .AddInput({norm_scale, ProgramTensorMetadataDependency::Type})
        .AddOutput({&normed, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(kRmsWorkgroupSize)
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(rows))
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(rows)},
                              {onnxruntime::narrow<uint32_t>(hc_mult)},
                              {onnxruntime::narrow<uint32_t>(hidden_size)},
                              {onnxruntime::narrow<uint32_t>(channels)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(norm_program));
  }

  // Pass 2: Convolution.
  if (total > 0) {
    const bool has_past = past_state != nullptr;
    const bool has_bias_input = bias != nullptr;
    ShortConvWithStateConvProgram conv_program{has_bias_input, has_past, activation_ == "silu" || activation_ == "swish"};
    conv_program.CacheHint(has_bias_input, has_past, activation_)
        .AddInput({&normed, ProgramTensorMetadataDependency::Type})
        .AddInput({weight, ProgramTensorMetadataDependency::Type});
    if (has_past) {
      conv_program.AddInput({past_state, ProgramTensorMetadataDependency::Type});
    }
    if (has_bias_input) {
      conv_program.AddInput({bias, ProgramTensorMetadataDependency::Type});
    }
    conv_program.AddOutput({output, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(hc_mult)},
                              {onnxruntime::narrow<uint32_t>(hidden_size)},
                              {onnxruntime::narrow<uint32_t>(kernel_size)},
                              {onnxruntime::narrow<uint32_t>(dilation_)},
                              {onnxruntime::narrow<uint32_t>(state_len)},
                              {onnxruntime::narrow<uint32_t>(channels)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(conv_program));
  }

  // Pass 3: Update state.
  const int64_t state_total = batch_size * channels * state_len;
  if (state_total > 0) {
    const bool has_past = past_state != nullptr;
    ShortConvWithStateUpdateProgram update_program{has_past};
    update_program.CacheHint(has_past)
        .AddInput({&normed, ProgramTensorMetadataDependency::Type});
    if (has_past) {
      update_program.AddInput({past_state, ProgramTensorMetadataDependency::Type});
    }
    update_program.AddOutput({present_state_out, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(state_total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(state_total)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(channels)},
                              {onnxruntime::narrow<uint32_t>(state_len)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(update_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
