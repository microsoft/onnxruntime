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
  F(bool, use_f32_accumulation_)               \
  F(InlinedVector<int64_t>, elements_per_thread_)

struct GemmSubgroupProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GEMM_SUBGROUP_PROGRAM_CONFIG);
  static Config MakeConfig(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul,
                           bool c_is_scalar, bool is_vec4, bool a_vec4, bool b_is_fp16, bool use_f32_accumulation,
                           const gsl::span<int64_t>& elements_per_thread) {
    Config result{};
    result.transA_ = bool{transA};
    result.transB_ = bool{transB};
    result.alpha_ = float{alpha};
    result.need_handle_bias_ = bool{need_handle_bias};
    result.need_handle_matmul_ = bool{need_handle_matmul};
    result.c_is_scalar_ = bool(c_is_scalar);
    result.is_vec4_ = bool(is_vec4);
    result.a_vec4_ = bool(a_vec4);
    result.b_is_fp16_ = bool(b_is_fp16);
    result.use_f32_accumulation_ = bool(use_f32_accumulation);
    result.elements_per_thread_ = InlinedVector<int64_t>(elements_per_thread.begin(), elements_per_thread.end());

    return result;
  }
  static constexpr std::string_view name = "GemmSubgroup";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GEMM_SUBGROUP_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, alpha)                                     \
  U(Float32, beta)                                      \
  U(Uint32, dim_a_outer)                                \
  U(Uint32, dim_b_outer)                                \
  U(Uint32, dim_inner)                                  \
  U(Uint32, logical_dispatch_x)                         \
  U(Uint32, logical_dispatch_y)                         \
  U(Uint32, logical_dispatch_z)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GEMM_SUBGROUP_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GEMM_SUBGROUP_PROGRAM_SHADER_UNIFORMS
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
