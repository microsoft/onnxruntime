// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <optional>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_CONFIG(F) \
  F(uint32_t, nbits_)                                           \
  F(SubgroupMatrixConfig, config_)                              \
  F(bool, has_zero_points_)                                     \
  F(bool, has_bias_)                                            \
  F(bool, has_weight_idx_)                                      \
  F(bool, has_weight_idx_indirect_)                             \
  F(uint32_t, tile_m_)                                          \
  F(uint32_t, tile_n_)                                          \
  F(bool, has_tail_buffer_)

struct SubgroupMatrixMatMulNBitsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t nbits, SubgroupMatrixConfig config, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect, uint32_t tile_m, uint32_t tile_n, bool has_tail_buffer) {
    Config result{};
    result.nbits_ = uint32_t(nbits);
    result.config_ = SubgroupMatrixConfig(config);
    result.has_zero_points_ = bool(has_zero_points);
    result.has_bias_ = bool(has_bias);
    result.has_weight_idx_ = bool{has_weight_idx};
    result.has_weight_idx_indirect_ = bool{has_weight_idx_indirect};
    result.tile_m_ = uint32_t(tile_m);
    result.tile_n_ = uint32_t(tile_n);
    result.has_tail_buffer_ = bool{has_tail_buffer};

    return result;
  }
  static constexpr std::string_view name = "SubgroupMatrixMatMulNBits";
  ;
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, M)                                                           \
  U(Uint32, N)                                                           \
  U(Uint32, K)                                                           \
  U(Uint32, zero_blocks_per_col)                                         \
  U(Uint32, num_N_tile)                                                  \
  U(Uint32, num_M_tile)                                                  \
  U(Uint32, weight_idx)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SUBGROUP_MATRIX_MAT_MUL_N_BITS_PROGRAM_CONFIG

using SubgroupMatrixMatMulNBitsProgram = ConfiguredProgram<SubgroupMatrixMatMulNBitsProgramShader>;

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      const Tensor* zero_points, const Tensor* bias,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t nbits,
                                      uint32_t zero_blocks_per_col,
                                      const SubgroupMatrixConfig& config,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y,
                                      const uint32_t weight_index,
                                      const Tensor* weight_index_indirect = nullptr);

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       uint32_t nbits,
                                       bool is_fp16,
                                       std::optional<SubgroupMatrixConfig>& config,
                                       uint32_t M = std::numeric_limits<uint32_t>::max(),
                                       bool has_weight_idx_indirect = false);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
