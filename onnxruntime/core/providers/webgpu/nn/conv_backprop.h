// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_CONFIG(F) \
  F(bool, is_channels_last_)                       \
  F(bool, has_bias_)                               \
  F(bool, is_prepacked_)                           \
  F(int, components_)

struct ConvTranspose3DProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_channels_last, bool has_bias, bool is_prepacked, int components) {
    Config result{};
    result.is_channels_last_ = bool(is_channels_last);
    result.has_bias_ = bool(has_bias);
    result.is_prepacked_ = bool(is_prepacked);
    result.components_ = int(components);

    return result;
  }
  static constexpr std::string_view name = "ConvTranspose3D";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                    \
  U(Uint32, strides)                                        \
  U(Uint32, dilations)                                      \
  U(Uint32, pads)                                           \
  U(Uint32, input_channels_per_group)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CONV_TRANSPOSE3_D_PROGRAM_CONFIG

using ConvTranspose3DProgram = ConfiguredProgram<ConvTranspose3DProgramShader>;

#define WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_CONFIG(F) \
  F(bool, is_channels_last_)                       \
  F(bool, has_bias_)                               \
  F(uint32_t, components_)                         \
  F(uint32_t, a_components_)                       \
  F(uint32_t, b_components_)                       \
  F(uint32_t, input_channels_remainder_)           \
  F(bool, pack_input_as4_)

struct ConvTranspose2DProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_channels_last, bool has_bias, uint32_t components, uint32_t a_components, uint32_t b_components, uint32_t input_channels_remainder, bool pack_input_as4) {
    Config result{};
    result.is_channels_last_ = bool(is_channels_last);
    result.has_bias_ = bool(has_bias);
    result.components_ = uint32_t(components);
    result.a_components_ = uint32_t(a_components);
    result.b_components_ = uint32_t(b_components);
    result.input_channels_remainder_ = uint32_t(input_channels_remainder);
    result.pack_input_as4_ = bool(pack_input_as4);

    return result;
  }
  static constexpr std::string_view name = "ConvTranspose2D";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                    \
  U(Uint32, strides)                                        \
  U(Uint32, filter_dims)                                    \
  U(Uint32, dilations)                                      \
  U(Uint32, effective_filter_dims)                          \
  U(Uint32, pads)                                           \
  U(Uint32, input_channels_per_group_int)                   \
  U(Uint32, input_channels_per_group)                       \
  U(Uint32, output_channels_per_group)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CONV_TRANSPOSE2_D_PROGRAM_CONFIG

using ConvTranspose2DProgram = ConfiguredProgram<ConvTranspose2DProgramShader>;

ConvTranspose2DProgram CreateConvTranspose2DProgram(const std::vector<const Tensor*>& inputs, const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides, const std::vector<uint32_t>& dilations, Tensor* output, bool is_channels_last, const std::vector<TensorShape>& modified_input_output_shapes, uint32_t groups);

}  // namespace webgpu
}  // namespace onnxruntime
