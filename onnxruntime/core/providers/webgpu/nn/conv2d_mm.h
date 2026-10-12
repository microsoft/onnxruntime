// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <string>
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {
#define WEBGPU_CONV2D_M_M_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)          \
  F(uint32_t, tile_inner_)                  \
  F(bool, fit_a_outer_)                     \
  F(bool, fit_b_outer_)                     \
  F(bool, fit_inner_)                       \
  F(bool, is_channels_last_)                \
  F(bool, is_vec4_)                         \
  F(bool, has_bias_)                        \
  F(std::vector<uint32_t>, element_size_)   \
  F(InlinedVector<int64_t>, elements_per_thread_)

struct Conv2dMMProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CONV2D_M_M_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation, uint32_t tile_inner, bool fit_a_outer, bool fit_b_outer, bool fit_inner,
                           bool is_channels_last, bool is_vec4, bool has_bias, std::vector<uint32_t>&& element_size,
                           InlinedVector<int64_t>&& elements_per_thread) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.tile_inner_ = uint32_t(tile_inner);
    result.fit_a_outer_ = bool(fit_a_outer);
    result.fit_b_outer_ = bool(fit_b_outer);
    result.fit_inner_ = bool(fit_inner);
    result.is_channels_last_ = bool(is_channels_last);
    result.is_vec4_ = bool(is_vec4);
    result.has_bias_ = bool(has_bias);
    result.element_size_ = std::vector<uint32_t>(std::move(element_size));
    result.elements_per_thread_ = InlinedVector<int64_t>(std::move(elements_per_thread));

    return result;
  }
  static constexpr std::string_view name = "Conv2dMM";
  static std::string Conv2dCommonSnippet(const Config& config, const ShaderVariableHelper& x,
                                         const ShaderVariableHelper& w, const ShaderActivation& activation,
                                         std::string data_type, uint32_t inner_element_size_x = 4,
                                         uint32_t inner_element_size_w = 4, uint32_t inner_element_size = 4);
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_CONV2D_M_M_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, dim_a_outer)                             \
  U(Uint32, dim_b_outer)                             \
  U(Uint32, dim_inner)                               \
  U(Uint32, pads)                                    \
  U(Uint32, strides)                                 \
  U(Uint32, dilations)                               \
  U(Uint32, logical_dispatch_x)                      \
  U(Uint32, logical_dispatch_y)                      \
  U(Uint32, logical_dispatch_z)                      \
  U(Float32, activation_param_0)                     \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CONV2D_M_M_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CONV2D_M_M_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CONV2D_M_M_PROGRAM_CONFIG

using Conv2dMMProgram = ConfiguredProgram<Conv2dMMProgramShader>;

Conv2dMMProgram CreateConv2dMMProgram(const Activation& activation, const std::vector<const Tensor*>& inputs, const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides, const std::vector<uint32_t>& dilations, Tensor* output, uint32_t dim_a_outer, uint32_t dim_b_outer, uint32_t dim_inner, bool is_channels_last, const std::vector<TensorShape>& modified_input_output_shapes);

}  // namespace webgpu
}  // namespace onnxruntime
