// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CONV3_D_NAIVE_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)             \
  F(bool, has_bias_)                           \
  F(bool, is_channels_last_)

struct Conv3DNaiveProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_CONV3_D_NAIVE_PROGRAM_CONFIG);
    Config(const Activation& activation, bool has_bias, bool is_channels_last)
        : activation_(activation), has_bias_(has_bias), is_channels_last_(is_channels_last) {}
  };
  static constexpr std::string_view name = "Conv3DNaive";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"filter_dims", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"x_spatial", ProgramUniformVariableDataType::Uint32},
      {"x_channels", ProgramUniformVariableDataType::Uint32},
      WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
};
#undef WEBGPU_CONV3_D_NAIVE_PROGRAM_CONFIG

using Conv3DNaiveProgram = ConfiguredProgram<Conv3DNaiveProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
