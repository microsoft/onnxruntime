// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/causal_conv_with_state.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

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
  shader.AddOutput("present_state", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/causal_conv_with_state.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_conv_state, has_conv_state_),
                             WGSL_TEMPLATE_PARAMETER(use_silu, activation_ == CausalConvActivation::Silu));
}

Status CausalConvWithState::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);       // (B, D, L)
  const Tensor* weight = context.Input(1);      // (D, 1, K)
  const Tensor* bias = context.Input(2);        // optional (D,)
  const Tensor* conv_state = context.Input(3);  // optional (B, D, K-1) — past_state

  ORT_RETURN_IF(activation_ == CausalConvActivation::Invalid, "Invalid activation type");
  ORT_RETURN_IF(ndim_ != 1, "Only 1D convolution is supported");
  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();

  ORT_RETURN_IF(input_shape.NumDimensions() != 3,
                "Input must be 3D (batch_size, channels, length)");
  ORT_RETURN_IF(weight_shape.NumDimensions() != 3,
                "Weight must be 3D (channels, 1, kernel_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t channels = input_shape[1];
  const int64_t input_length = input_shape[2];
  const int64_t kernel_size = weight_shape[2];
  const int64_t state_length = kernel_size - 1;

  ORT_RETURN_IF(weight_shape[0] != channels, "Weight first dim must match input channels");
  ORT_RETURN_IF(weight_shape[1] != 1, "Weight second dim must be 1 for depthwise convolution");

  if (bias != nullptr) {
    ORT_RETURN_IF(bias->Shape().NumDimensions() != 1, "Bias must be 1D");
    ORT_RETURN_IF(bias->Shape()[0] != channels, "Bias size must match channels");
  }

  if (conv_state != nullptr) {
    ORT_RETURN_IF(conv_state->Shape().NumDimensions() != 3,
                  "conv_state must be 3D (batch_size, channels, kernel_size - 1)");
    ORT_RETURN_IF(conv_state->Shape()[0] != batch_size,
                  "conv_state batch_size must match input");
    ORT_RETURN_IF(conv_state->Shape()[1] != channels,
                  "conv_state channels must match input");
    ORT_RETURN_IF(conv_state->Shape()[2] != state_length,
                  "conv_state last dim must be kernel_size - 1");
  }

  const bool has_bias = (bias != nullptr);
  const bool has_conv_state = (conv_state != nullptr);

  // Allocate outputs
  // Output 0: (B, D, L)
  Tensor* output = context.Output(0, input_shape);

  // Output 1: present_state (B, D, K-1)
  std::vector<int64_t> state_dims{batch_size, channels, state_length};
  Tensor* present_state = context.Output(1, TensorShape(state_dims));

  if (input_shape.Size() == 0) {
    if (has_conv_state) {
      ORT_RETURN_IF_ERROR(context.CopyTensor(*conv_state, *present_state));
    } else {
      context.FillZero(*present_state);
      return Status::OK();
    }
  }

  // Create and run the shader program
  CausalConvWithStateProgram program{activation_, has_bias, has_conv_state};

  uint32_t output_size = static_cast<uint32_t>(batch_size * channels * input_length);

  program.CacheHint(has_bias, has_conv_state, kernel_size, static_cast<int>(activation_));

  program.AddInput({input, ProgramTensorMetadataDependency::Type})
      .AddInput({weight, ProgramTensorMetadataDependency::None});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_conv_state) {
    program.AddInput({conv_state, ProgramTensorMetadataDependency::None});
  }

  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .AddOutput({present_state, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariable({static_cast<uint32_t>(batch_size)})
      .AddUniformVariable({static_cast<uint32_t>(channels)})
      .AddUniformVariable({static_cast<uint32_t>(input_length)})
      .AddUniformVariable({static_cast<uint32_t>(kernel_size)})
      .AddUniformVariable({static_cast<uint32_t>(state_length)})
      .AddUniformVariable({output_size});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
