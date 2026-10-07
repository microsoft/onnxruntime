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
  F(uint32_t, split_dim_inner_)

struct MatMulProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MAT_MUL_PROGRAM_CONFIG);
    Config(const Activation& activation, bool bias, bool is_vec4, const gsl::span<int64_t>& elements_per_thread,
           bool is_channels_last = false, uint32_t split_dim_inner = 1)
        : activation_(activation),
          has_bias_{bias},
          is_vec4_{is_vec4},
          elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()),
          is_channels_last_(is_channels_last),
          split_dim_inner_(split_dim_inner) {}
  };
  static constexpr std::string_view name = "MatMul";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_x", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_y", ProgramUniformVariableDataType::Uint32},
                                          {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32},
                                          {"splits_per_batch", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_CONFIG);
    Config(bool is_gemm, bool has_bias, uint32_t output_components, bool bias_is_scalar)
        : is_gemm_(is_gemm),
          has_bias_(has_bias),
          output_components_(output_components),
          bias_is_scalar_(bias_is_scalar) {}
  };
  static constexpr std::string_view name = "MatMul_Fill_Bias_Or_Zero_Before_Split_K";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"beta", ProgramUniformVariableDataType::Float32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MAT_MUL_FILL_BIAS_OR_ZERO_BEFORE_SPLIT_K_PROGRAM_CONFIG

using MatMulFillBiasOrZeroBeforeSplitKProgram = ConfiguredProgram<MatMulFillBiasOrZeroBeforeSplitKProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
