// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/webgpu/math/gemm.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Creates a GemmOptImpl that runs the subgroup-matrix kernel on devices whose
// vendor policy supports it. The per-problem output tiling comes from a
// vendor-specific selector chosen internally from the device context (the same
// selector used by the subgroup-matrix MatMul). Returns nullptr when no vendor
// policy applies, so the caller falls back to the default Gemm path.
std::unique_ptr<Gemm::GemmOptImpl> CreateSubgroupMatrixGemmImpl(
    const Gemm& parent, const ComputeContextBase& context);

// Computes Y = alpha * op(A) @ op(B) + beta * C using subgroupMatrixMultiplyAccumulate.
// config selects the device-supported subgroup-matrix configuration; sg_mat_count_m/n select how
// many subgroup matrices the tile spans along M/N; split_k is the number of subgroups that
// cooperatively reduce the K dimension. trans_a / trans_b select the A / B load
// majorness; has_c enables the beta * C epilogue (C broadcast to [M, N] via the
// c_stride_m / c_stride_n uniforms).
#define WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_CONFIG(F) \
  F(bool, has_c_)                                     \
  F(bool, trans_a_)                                   \
  F(bool, trans_b_)                                   \
  F(SubgroupMatrixConfig, config_)                    \
  F(uint32_t, sg_mat_count_m_)                        \
  F(uint32_t, sg_mat_count_n_)                        \
  F(uint32_t, split_k_)

struct SubgroupMatrixGemmProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_c, bool trans_a, bool trans_b, SubgroupMatrixConfig config, uint32_t sg_mat_count_m,
                           uint32_t sg_mat_count_n, uint32_t split_k) {
    Config result{};
    result.has_c_ = bool(has_c);
    result.trans_a_ = bool(trans_a);
    result.trans_b_ = bool(trans_b);
    result.config_ = SubgroupMatrixConfig(config);
    result.sg_mat_count_m_ = uint32_t(sg_mat_count_m);
    result.sg_mat_count_n_ = uint32_t(sg_mat_count_n);
    result.split_k_ = uint32_t(split_k);

    return result;
  }
  static constexpr std::string_view name = "SubgroupMatrixGemm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, M)                                                 \
  U(Uint32, N)                                                 \
  U(Uint32, K)                                                 \
  U(Float32, alpha)                                            \
  U(Float32, beta)                                             \
  U(Uint32, c_stride_m)                                        \
  U(Uint32, c_stride_n)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SUBGROUP_MATRIX_GEMM_PROGRAM_CONFIG

using SubgroupMatrixGemmProgram = ConfiguredProgram<SubgroupMatrixGemmProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
