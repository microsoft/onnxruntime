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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CONV3_D_NAIVE_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation, bool has_bias, bool is_channels_last) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.has_bias_ = bool(has_bias);
    result.is_channels_last_ = bool(is_channels_last);

    return result;
  }
  static constexpr std::string_view name = "Conv3DNaive";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_CONV3_D_NAIVE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                \
  U(Uint32, filter_dims)                                \
  U(Uint32, pads)                                       \
  U(Uint32, strides)                                    \
  U(Uint32, dilations)                                  \
  U(Uint32, x_spatial)                                  \
  U(Uint32, x_channels)                                 \
  U(Float32, activation_param_0)                        \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CONV3_D_NAIVE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CONV3_D_NAIVE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CONV3_D_NAIVE_PROGRAM_CONFIG

using Conv3DNaiveProgram = ConfiguredProgram<Conv3DNaiveProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
