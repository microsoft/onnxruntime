// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/causal_conv_with_state.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"

using namespace onnxruntime::webgpu;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

CausalConvActivation ParseCausalConvActivation(const std::string& activation_str) {
  if (activation_str == "silu" || activation_str == "swish") {
    return CausalConvActivation::Silu;
  } else if (activation_str == "none" || activation_str.empty()) {
    return CausalConvActivation::None;
  }
  return CausalConvActivation::Invalid;
}

// =============================================================================
// CausalConvWithState Implementation
// =============================================================================

ONNX_OPERATOR_KERNEL_EX(
    CausalConvWithState,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    CausalConvWithState);

CausalConvWithState::CausalConvWithState(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string activation_str = info.GetAttrOrDefault<std::string>("activation", "none");
  activation_ = ParseCausalConvActivation(activation_str);
  ORT_ENFORCE(info.GetAttr<int64_t>("ndim", &ndim_).IsOK(), "Attribute 'ndim' is required");
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseDilation(info, dilation_));
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseChannelsLast(info, channels_last_));
  ORT_ENFORCE(!channels_last_ || ndim_ == 1, "channels_last requires ndim = 1");
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("state_window", 0) == 0,
              "WebGPU CausalConvWithState does not support state_window > 0 (CUDA EP only)");
}

Status CausalConvWithStateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("weight", ShaderUsage::UseUniform);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_conv_state_) {
    shader.AddInput("conv_state", ShaderUsage::UseUniform);
  }

  shader.AddOutput("output", ShaderUsage::UseUniform);
  if (output_present_state_) {
    shader.AddOutput("present_state", ShaderUsage::UseUniform);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/causal_conv_with_state.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(channels_last, channels_last_),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_conv_state, has_conv_state_),
                             WGSL_TEMPLATE_PARAMETER(output_present_state, output_present_state_),
                             WGSL_TEMPLATE_PARAMETER(use_silu, activation_ == CausalConvActivation::Silu));
}

Status CausalConvUpdateStateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("present_state", ShaderUsage::UseUniform);

  // global_idx enumerates (batch, channel) pairs. Both layouts are dense, so a (base, stride)
  // pair per tensor covers them: channels-first walks a contiguous row, channels-last strides by
  // `channels`.
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.update_size");
  if (channels_last_) {
    shader.MainFunctionBody()
        << "  let batch_idx = global_idx / uniforms.channels;\n"
           "  let channel_idx = global_idx % uniforms.channels;\n"
           "  let base_state = batch_idx * uniforms.channels * uniforms.state_length + channel_idx;\n"
           "  let state_stride = uniforms.channels;\n"
           "  let base_input = batch_idx * uniforms.channels * uniforms.input_length + channel_idx;\n"
           "  let input_stride = uniforms.channels;\n";
  } else {
    shader.MainFunctionBody()
        << "  let base_state = global_idx * uniforms.state_length;\n"
           "  let state_stride = 1u;\n"
           "  let base_input = global_idx * uniforms.input_length;\n"
           "  let input_stride = 1u;\n";
  }

  shader.MainFunctionBody()
      << "\n"
         "  if (uniforms.input_length >= uniforms.state_length) {\n"
         "    let input_offset = uniforms.input_length - uniforms.state_length;\n"
         "    for (var s = 0u; s < uniforms.state_length; s++) {\n"
         "      present_state[base_state + s * state_stride] =\n"
         "          input[base_input + (input_offset + s) * input_stride];\n"
         "    }\n"
         "  } else {\n"
         "    let preserved_state = uniforms.state_length - uniforms.input_length;\n"
         "    for (var s = 0u; s < uniforms.state_length; s++) {\n"
         "      if (s < preserved_state) {\n"
         "        present_state[base_state + s * state_stride] =\n"
         "            present_state[base_state + (s + uniforms.input_length) * state_stride];\n"
         "      } else {\n"
         "        present_state[base_state + s * state_stride] =\n"
         "            input[base_input + (s - preserved_state) * input_stride];\n"
         "      }\n"
         "    }\n"
         "  }\n";

  return Status::OK();
}

Status CausalConvWithState::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);       // (B, D, L)
  const Tensor* weight = context.Input(1);      // (D, 1, K)
  const Tensor* bias = context.Input(2);        // optional (D,)
  const Tensor* conv_state = context.Input(3);  // optional (B, D, (K-1)*dilation) — past_state

  ORT_RETURN_IF(activation_ == CausalConvActivation::Invalid, "Invalid activation type");
  ORT_RETURN_IF(ndim_ != 1, "Only 1D convolution is supported");
  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();

  if (channels_last_) {
    ORT_RETURN_IF(input_shape.NumDimensions() < 3,
                  "Input must have rank >= 3 (batch_size, sequence_length, ...channels) when channels_last = 1");
  } else {
    ORT_RETURN_IF(input_shape.NumDimensions() != 3,
                  "Input must be 3D (batch_size, channels, length)");
  }
  ORT_RETURN_IF(weight_shape.NumDimensions() != 3,
                "Weight must be 3D (channels, 1, kernel_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t channels = channels_last_ ? input_shape.SizeFromDimension(2) : input_shape[1];
  const int64_t input_length = channels_last_ ? input_shape[1] : input_shape[2];
  const int64_t kernel_size = weight_shape[2];
  const int64_t state_length = (kernel_size - 1) * dilation_;

  ORT_RETURN_IF(weight_shape[0] != channels, "Weight first dim must match input channels");
  ORT_RETURN_IF(weight_shape[1] != 1, "Weight second dim must be 1 for depthwise convolution");

  if (bias != nullptr) {
    ORT_RETURN_IF(bias->Shape().NumDimensions() != 1, "Bias must be 1D");
    ORT_RETURN_IF(bias->Shape()[0] != channels, "Bias size must match channels");
  }

  TensorShapeVector state_dims;
  if (channels_last_) {
    state_dims.push_back(batch_size);
    state_dims.push_back(state_length);
    for (size_t i = 2; i < input_shape.NumDimensions(); ++i) {
      state_dims.push_back(input_shape[i]);
    }
  } else {
    state_dims = TensorShapeVector{batch_size, channels, state_length};
  }
  const TensorShape state_shape(state_dims);

  if (conv_state != nullptr) {
    ORT_RETURN_IF(conv_state->Shape() != state_shape,
                  "conv_state is expected to have shape ", state_shape.ToString(),
                  ", got ", conv_state->Shape().ToString());
  }

  const bool has_bias = (bias != nullptr);
  const bool has_conv_state = (conv_state != nullptr);

  // Allocate outputs
  // Output 0: (B, D, L)
  Tensor* output = context.Output(0, input_shape);

  // Output 1: present_state, matching the layout selected by channels_last
  Tensor* present_state = context.Output(1, state_shape);
  const bool conv_state_in_present_state = has_conv_state && conv_state->DataRaw() == present_state->DataRaw();

  if (input_shape.Size() == 0) {
    if (has_conv_state) {
      if (!conv_state_in_present_state) {
        ORT_RETURN_IF_ERROR(context.CopyTensor(*conv_state, *present_state));
      }
    } else {
      context.FillZero(*present_state);
    }
    return Status::OK();
  }

  // Create and run the shader program
  CausalConvWithStateProgram program{activation_, has_bias, has_conv_state, !conv_state_in_present_state,
                                     channels_last_};

  uint32_t output_size = static_cast<uint32_t>(batch_size * channels * input_length);

  program.CacheHint(has_bias, has_conv_state, !conv_state_in_present_state,
                    kernel_size, dilation_, static_cast<int>(activation_), channels_last_);

  program.AddInput({input, ProgramTensorMetadataDependency::Type})
      .AddInput({weight, ProgramTensorMetadataDependency::None});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_conv_state) {
    program.AddInput({conv_state, ProgramTensorMetadataDependency::None});
  }

  program.AddOutput({output, ProgramTensorMetadataDependency::None});
  if (!conv_state_in_present_state) {
    program.AddOutput({present_state, ProgramTensorMetadataDependency::None});
  }

  program.SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariable({static_cast<uint32_t>(batch_size)})
      .AddUniformVariable({static_cast<uint32_t>(channels)})
      .AddUniformVariable({static_cast<uint32_t>(input_length)})
      .AddUniformVariable({static_cast<uint32_t>(kernel_size)})
      .AddUniformVariable({static_cast<uint32_t>(dilation_)})
      .AddUniformVariable({static_cast<uint32_t>(state_length)})
      .AddUniformVariable({output_size});

  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  if (conv_state_in_present_state) {
    CausalConvUpdateStateProgram update_state_program{channels_last_};
    const uint32_t update_size = static_cast<uint32_t>(batch_size * channels);
    update_state_program.CacheHint(channels_last_);
    update_state_program.AddInput({input, ProgramTensorMetadataDependency::Type})
        .AddOutput({present_state, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((update_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{static_cast<uint32_t>(channels)},
                              {static_cast<uint32_t>(input_length)},
                              {static_cast<uint32_t>(state_length)},
                              {update_size}});

    ORT_RETURN_IF_ERROR(context.RunProgram(update_state_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
