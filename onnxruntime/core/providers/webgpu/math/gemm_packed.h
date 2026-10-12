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
  F(uint32_t, split_dim_inner_)       \
  F(bool, use_f32_accumulation_)

struct GemmProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GEMM_PROGRAM_CONFIG);
  static Config MakeConfig(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul,
                           bool c_is_scalar, int output_components, bool is_vec4 = false,
                           uint32_t split_dim_inner = 1, bool use_f32_accumulation = false) {
    Config result{};
    result.transA_ = bool{transA};
    result.transB_ = bool{transB};
    result.alpha_ = float{alpha};
    result.need_handle_bias_ = bool{need_handle_bias};
    result.need_handle_matmul_ = bool{need_handle_matmul};
    result.c_is_scalar_ = bool(c_is_scalar);
    result.output_components_ = int(output_components);
    result.is_vec4_ = bool(is_vec4);
    result.split_dim_inner_ = uint32_t(split_dim_inner);
    result.use_f32_accumulation_ = bool(use_f32_accumulation);

    return result;
  }
  static constexpr std::string_view name = "Gemm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GEMM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, alpha)                            \
  U(Float32, beta)                             \
  U(Uint32, dim_a_outer)                       \
  U(Uint32, dim_b_outer)                       \
  U(Uint32, dim_inner)                         \
  U(Uint32, logical_dispatch_x)                \
  U(Uint32, logical_dispatch_y)                \
  U(Uint32, logical_dispatch_z)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GEMM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GEMM_PROGRAM_SHADER_UNIFORMS

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
