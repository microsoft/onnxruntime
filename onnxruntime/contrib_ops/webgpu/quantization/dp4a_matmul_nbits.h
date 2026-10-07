// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_D_P4_A_MAT_MUL_QUANTIZE_PROGRAM_CONFIG(F)

struct DP4AMatMulQuantizeProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_D_P4_A_MAT_MUL_QUANTIZE_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "DP4AMatMulQuantize";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_D_P4_A_MAT_MUL_QUANTIZE_PROGRAM_CONFIG

using DP4AMatMulQuantizeProgram = ConfiguredProgram<DP4AMatMulQuantizeProgramShader>;

#define WEBGPU_D_P4_A_MAT_MUL_N_BITS_PROGRAM_CONFIG(F) \
  F(uint32_t, block_size_)                             \
  F(uint32_t, nbits_)                                  \
  F(bool, has_bias_)                                   \
  F(bool, has_zero_points_)                            \
  F(bool, has_weight_idx_)                             \
  F(bool, has_weight_idx_indirect_)                    \
  F(bool, is_qualcomm_)                                \
  F(bool, acc_f32_)

struct DP4AMatMulNBitsProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_D_P4_A_MAT_MUL_N_BITS_PROGRAM_CONFIG);
    Config(uint32_t block_size, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx,
           bool has_weight_idx_indirect, bool is_qualcomm, bool acc_f32)
        : block_size_(block_size),
          nbits_(nbits),
          has_bias_(has_bias),
          has_zero_points_(has_zero_points),
          has_weight_idx_(has_weight_idx),
          has_weight_idx_indirect_(has_weight_idx_indirect),
          is_qualcomm_(is_qualcomm),
          acc_f32_(acc_f32) {}
  };
  static constexpr std::string_view name = "DP4AMatMulNBits";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K8", ProgramUniformVariableDataType::Uint32},
      {"K16", ProgramUniformVariableDataType::Uint32},
      {"num_M_tile", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_D_P4_A_MAT_MUL_N_BITS_PROGRAM_CONFIG

using DP4AMatMulNBitsProgram = ConfiguredProgram<DP4AMatMulNBitsProgramShader>;

#define WEBGPU_D_P4_A_MAT_MUL_N_BITS_SMALL_M_PROGRAM_CONFIG(F) \
  F(uint32_t, tile_size_k_vec_)                                \
  F(uint32_t, tile_size_)                                      \
  F(uint32_t, nbits_)                                          \
  F(bool, has_bias_)                                           \
  F(bool, has_zero_points_)                                    \
  F(bool, has_weight_idx_)                                     \
  F(bool, has_weight_idx_indirect_)                            \
  F(bool, single_scale_weights_)                               \
  F(bool, broadcast_a_row_)                                    \
  F(bool, acc_f32_)

struct DP4AMatMulNBitsSmallMProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_D_P4_A_MAT_MUL_N_BITS_SMALL_M_PROGRAM_CONFIG);
    Config(uint32_t tile_size_k_vec, uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias,
           bool has_weight_idx, bool has_weight_idx_indirect, bool single_scale_weights, bool broadcast_a_row = false,
           bool acc_f32 = false)
        : tile_size_k_vec_(tile_size_k_vec),
          tile_size_(tile_size),
          nbits_(nbits),
          has_bias_(has_bias),
          has_zero_points_(has_zero_points),
          has_weight_idx_(has_weight_idx),
          has_weight_idx_indirect_(has_weight_idx_indirect),
          single_scale_weights_(single_scale_weights),
          broadcast_a_row_(broadcast_a_row),
          acc_f32_(acc_f32) {}
  };
  static constexpr std::string_view name = "DP4AMatMulNBitsSmallMProgram";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K16", ProgramUniformVariableDataType::Uint32},
      {"K32", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"dispatch_M", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_D_P4_A_MAT_MUL_N_BITS_SMALL_M_PROGRAM_CONFIG

using DP4AMatMulNBitsSmallMProgram = ConfiguredProgram<DP4AMatMulNBitsSmallMProgramShader>;

Status ApplyDP4AMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points, const Tensor* bias,
                                  uint32_t batch_count,
                                  uint32_t M,
                                  uint32_t dispatch_M,
                                  uint32_t N,
                                  uint32_t K,
                                  uint32_t block_size,
                                  uint32_t zero_blocks_per_col,
                                  uint32_t min_M_for_tile_optimization,
                                  uint32_t nbits,
                                  onnxruntime::webgpu::ComputeContext& context,
                                  Tensor* y,
                                  const uint32_t weight_index,
                                  const Tensor* weight_index_indirect = nullptr);

// The optional M / has_weight_idx_indirect / y arguments fold the original
// dispatch-precondition (DP4A is preferred when M is large enough, or
// unconditionally on FP32 outputs and Qualcomm GPUs) into the feasibility check
// so callers don't need a separate wrapper. Defaults make the precondition
// trivially satisfied for callers that only want the feasibility check.
bool CanApplyDP4AMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                   uint64_t accuracy_level,
                                   uint32_t block_size,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t components_k,
                                   uint32_t M = std::numeric_limits<uint32_t>::max(),
                                   bool has_weight_idx_indirect = false,
                                   const Tensor* y = nullptr);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
