// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GROUPED_CONV_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)            \
  F(bool, has_bias_)                          \
  F(bool, is_channels_last_)                  \
  F(bool, depthwise_vec_)

struct GroupedConvProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GROUPED_CONV_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation, bool has_bias, bool is_channels_last, bool depthwise_vec = false) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.has_bias_ = bool(has_bias);
    result.is_channels_last_ = bool(is_channels_last);
    result.depthwise_vec_ = bool(depthwise_vec);

    return result;
  }
  static constexpr std::string_view name = "GroupedConv";
  // depthwise_vec selects the NHWC depthwise form, in which every group has exactly one output
  // channel so input and output channels correspond 1:1. That lets x, w and the output all be
  // indexed with the same vectorized channel index, which the general grouped path cannot do
  // because its input channels do not line up with output vectors.

  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_GROUPED_CONV_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                               \
  U(Uint32, dilations)                                 \
  U(Uint32, strides)                                   \
  U(Uint32, pads)                                      \
  U(Uint32, output_channels_per_group)                 \
  U(Uint32, components)                                \
  U(Float32, activation_param_0)                       \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GROUPED_CONV_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GROUPED_CONV_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GROUPED_CONV_PROGRAM_CONFIG

using GroupedConvProgram = ConfiguredProgram<GroupedConvProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
