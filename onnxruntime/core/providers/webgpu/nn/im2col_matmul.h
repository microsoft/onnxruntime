// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                             \
  F(uint32_t, tile_m_)                           \
  F(uint32_t, tile_n_)                           \
  F(uint32_t, vec_size_)                         \
  F(bool, use_subgroup_)                         \
  F(ShaderActivation, activation_)

struct Im2ColMatMulProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias, uint32_t tile_m, uint32_t tile_n, uint32_t vec_size, bool use_subgroup,
                           const Activation& activation) {
    Config result{};
    result.has_bias_ = bool(has_bias);
    result.tile_m_ = uint32_t(tile_m);
    result.tile_n_ = uint32_t(tile_n);
    result.vec_size_ = uint32_t(vec_size);
    result.use_subgroup_ = bool(use_subgroup);
    result.activation_ = MakeShaderActivation(activation);

    return result;
  }
  static constexpr std::string_view name = "Im2ColMatMul";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_IM2_COL_MAT_MUL_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch)                                        \
  U(Uint32, src_h)                                        \
  U(Uint32, src_w)                                        \
  U(Uint32, channel_i)                                    \
  U(Uint32, kernel_h)                                     \
  U(Uint32, kernel_w)                                     \
  U(Uint32, output_h)                                     \
  U(Uint32, output_w)                                     \
  U(Uint32, im2col_m)                                     \
  U(Uint32, im2col_k)                                     \
  U(Uint32, im2col_n)                                     \
  U(Uint32, M_tiles)                                      \
  U(Uint32, N_tiles)                                      \
  U(Uint32, K_tiles)                                      \
  U(Uint32, dilations)                                    \
  U(Uint32, pads)                                         \
  U(Uint32, strides)                                      \
  U(Float32, activation_param_0)                          \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_IM2_COL_MAT_MUL_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_IM2_COL_MAT_MUL_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG

using Im2ColMatMulProgram = ConfiguredProgram<Im2ColMatMulProgramShader>;

bool CanApplyIm2ColMatMulProgram(ComputeContextBase& context,
                                 const bool is_channels_last,
                                 const Activation& activation,
                                 const TensorShape kernel_shape,
                                 const uint32_t group,
                                 const MLDataType data_type);

// Transposes the OIHW weight into the OHWI layout expected by Im2ColMatMulProgram.
// Called from Conv::PrePackInternal so the transpose runs once at session
// initialization instead of on every inference.
Status PrePackIm2ColMatMulWeight(ComputeContextBase& context,
                                 const Tensor& weight,
                                 AllocatorPtr alloc,
                                 /*out*/ std::unique_ptr<Tensor>& packed_weight);

// `packed_weight` is the OHWI weight produced by PrePackIm2ColMatMulWeight. When it
// is nullptr, the OIHW weight is read from input 1 and transposed on the fly.
Status ApplyIm2ColMatMulProgram(ComputeContext& context,
                                const bool is_channels_last,
                                const Activation& activation,
                                const std::vector<uint32_t>& dilations,
                                const std::vector<uint32_t>& pads,
                                const std::vector<uint32_t>& strides,
                                const Tensor* packed_weight,
                                Tensor* output);

}  // namespace webgpu
}  // namespace onnxruntime
