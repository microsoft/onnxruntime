// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GEMM_PROGRAM_CONFIG(F) \
  F(bool, transA_)                    \
  F(bool, transB_)                    \
  F(float, alpha_)                    \
  F(bool, need_handle_bias_)          \
  F(bool, need_handle_matmul_)        \
  F(bool, c_is_scalar_)               \
  F(int, output_components_)          \
  F(bool, is_vec4_)                   \
  F(uint32_t, split_dim_inner_)

struct GemmProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GEMM_PROGRAM_CONFIG);
    Config(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul, bool c_is_scalar,
           int output_components, bool is_vec4 = false, uint32_t split_dim_inner = 1)
        : transA_{transA},
          transB_{transB},
          alpha_{alpha},
          need_handle_bias_{need_handle_bias},
          need_handle_matmul_{need_handle_matmul},
          c_is_scalar_(c_is_scalar),
          output_components_(output_components),
          is_vec4_(is_vec4),
          split_dim_inner_(split_dim_inner) {}
  };
  static constexpr std::string_view name = "Gemm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"dim_a_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_inner", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_x", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_y", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32});

  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_X = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Y = 8;
  constexpr static uint32_t MATMUL_PACKED_WORKGROUP_SIZE_Z = 1;
};
#undef WEBGPU_GEMM_PROGRAM_CONFIG

using GemmProgram = ConfiguredProgram<GemmProgramShader>;

Status ApplyGemmPacked(const Tensor* a,
                       const Tensor* b,
                       const Tensor* c,
                       bool transA,
                       bool transB,
                       float alpha,
                       float beta,
                       ComputeContext& context);

}  // namespace webgpu
}  // namespace onnxruntime
