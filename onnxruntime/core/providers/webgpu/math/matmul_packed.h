// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
#define WEBGPU_MAT_MUL_PROGRAM_CONFIG(F)          \
  F(ShaderActivation, activation_)                \
  F(bool, has_bias_)                              \
  F(bool, is_vec4_)                               \
  F(InlinedVector<int64_t>, elements_per_thread_) \
  F(bool, is_channels_last_)                      \
  F(uint32_t, split_dim_inner_)                   \
  F(uint32_t, tile_inner_)                        \
  F(bool, use_f32_accumulation_)

struct MatMulProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_PROGRAM_CONFIG);
  static Config MakeConfig(const Activation& activation,
                           bool bias,
                           bool is_vec4,
                           const gsl::span<int64_t>& elements_per_thread,
                           bool is_channels_last = false,
                           uint32_t split_dim_inner = 1,
                           uint32_t tile_inner = 32,
                           bool use_f32_accumulation = false) {
    Config result{};
    result.activation_ = MakeShaderActivation(activation);
    result.has_bias_ = bool{bias};
    result.is_vec4_ = bool{is_vec4};
    result.elements_per_thread_ = InlinedVector<int64_t>(elements_per_thread.begin(), elements_per_thread.end());
    result.is_channels_last_ = bool(is_channels_last);
    result.split_dim_inner_ = uint32_t(split_dim_inner);
    result.tile_inner_ = uint32_t(tile_inner);
    result.use_f32_accumulation_ = bool(use_f32_accumulation);

    return result;
  }
  static constexpr std::string_view name = "MatMul";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_MAT_MUL_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, dim_a_outer)                          \
  U(Uint32, dim_b_outer)                          \
  U(Uint32, dim_inner)                            \
  U(Uint32, logical_dispatch_x)                   \
  U(Uint32, logical_dispatch_y)                   \
  U(Uint32, logical_dispatch_z)                   \
  U(Uint32, splits_per_batch)                     \
  U(Float32, activation_param_0)                  \
  U(Float32, activation_param_1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_PROGRAM_CONFIG

using MatMulProgram = ConfiguredProgram<MatMulProgramShader>;

// The program to initialize the output with 0 or bias before doing MatMul with Split-K. In Split-K,
// we set the output values with `atomicLoad` and `atomicCompareExchangeWeak` instead of a direct
// assignment (see the function `HandleMatMulWithSplitK()` in `gemm_utils.cc`), so we must initialize
// the output with 0 or bias first to make sure `atomicLoad` won't return garbage data.
#define WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_CONFIG(F) \
  F(bool, is_gemm_)                                                       \
  F(bool, has_bias_)                                                      \
  F(uint32_t, output_components_)                                         \
  F(bool, bias_is_scalar_)

struct MatMulFillBiasOrZeroBeforeSplitKProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_gemm, bool has_bias, uint32_t output_components, bool bias_is_scalar) {
    Config result{};
    result.is_gemm_ = bool(is_gemm);
    result.has_bias_ = bool(has_bias);
    result.output_components_ = uint32_t(output_components);
    result.bias_is_scalar_ = bool(bias_is_scalar);

    return result;
  }
  static constexpr std::string_view name = "MatMul_Fill_Bias_Or_Zero_Before_Split_K";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, dim_a_outer)                                                           \
  U(Uint32, dim_b_outer)                                                           \
  U(Float32, beta)                                                                 \
  U(Uint32, batch_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_CONFIG

using MatMulFillBiasOrZeroBeforeSplitKProgram = ConfiguredProgram<MatMulFillBiasOrZeroBeforeSplitKProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
