// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

#define WEBGPU_GEMM_SUBGROUP_PROGRAM_CONFIG(F) \
  F(bool, transA_)                             \
  F(bool, transB_)                             \
  F(float, alpha_)                             \
  F(bool, need_handle_bias_)                   \
  F(bool, need_handle_matmul_)                 \
  F(bool, c_is_scalar_)                        \
  F(bool, is_vec4_)                            \
  F(bool, a_vec4_)                             \
  F(bool, b_is_fp16_)                          \
  F(InlinedVector<int64_t>, elements_per_thread_)

struct GemmSubgroupProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GEMM_SUBGROUP_PROGRAM_CONFIG);
    Config(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul, bool c_is_scalar,
           bool is_vec4, bool a_vec4, bool b_is_fp16, const gsl::span<int64_t>& elements_per_thread)
        : transA_{transA},
          transB_{transB},
          alpha_{alpha},
          need_handle_bias_{need_handle_bias},
          need_handle_matmul_{need_handle_matmul},
          c_is_scalar_(c_is_scalar),
          is_vec4_(is_vec4),
          a_vec4_(a_vec4),
          b_is_fp16_(b_is_fp16),
          elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()) {}
  };
  static constexpr std::string_view name = "GemmSubgroup";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"dim_a_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_inner", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GEMM_SUBGROUP_PROGRAM_CONFIG

using GemmSubgroupProgram = ConfiguredProgram<GemmSubgroupProgramShader>;

bool CanApplyGemmIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA, bool transB);

Status ApplyGemmIntel(const Tensor* a,
                      const Tensor* b,
                      const Tensor* c,
                      bool transA,
                      bool transB,
                      float alpha,
                      float beta,
                      ComputeContext& context);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
