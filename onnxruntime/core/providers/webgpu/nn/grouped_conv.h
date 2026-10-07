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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GROUPED_CONV_PROGRAM_CONFIG);
    Config(const Activation& activation, bool has_bias, bool is_channels_last, bool depthwise_vec = false)
        : activation_(activation),
          has_bias_(has_bias),
          is_channels_last_(is_channels_last),
          depthwise_vec_(depthwise_vec) {}
  };
  static constexpr std::string_view name = "GroupedConv";
  // depthwise_vec selects the NHWC depthwise form, in which every group has exactly one output
  // channel so input and output channels correspond 1:1. That lets x, w and the output all be
  // indexed with the same vectorized channel index, which the general grouped path cannot do
  // because its input channels do not line up with output vectors.

  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"output_channels_per_group", ProgramUniformVariableDataType::Uint32},
      {"components", ProgramUniformVariableDataType::Uint32},
      WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
};
#undef WEBGPU_GROUPED_CONV_PROGRAM_CONFIG

using GroupedConvProgram = ConfiguredProgram<GroupedConvProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
