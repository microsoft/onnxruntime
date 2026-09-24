// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <optional>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SubgroupMatrixMatMulNBitsProgram final : public Program<SubgroupMatrixMatMulNBitsProgram> {
 public:
  SubgroupMatrixMatMulNBitsProgram(uint32_t nbits, SubgroupMatrixConfig config, uint32_t tile_size_m,
                                   uint32_t tile_size_n, uint32_t tile_size_k,
                                   bool has_zero_points, bool has_bias,
                                   bool has_weight_idx, bool has_weight_idx_indirect, bool has_tail_buffer)
      : Program{config.subgroupSize == 64 ? "SubgroupMatrixMatMulNBitsWave64" : "SubgroupMatrixMatMulNBits"},
        nbits_(nbits),
        config_(config),
        tile_size_m_(tile_size_m),
        tile_size_n_(tile_size_n),
        tile_size_k_(tile_size_k),
        has_zero_points_(has_zero_points),
        has_bias_(has_bias),
        has_weight_idx_{has_weight_idx},
        has_weight_idx_indirect_{has_weight_idx_indirect},
        has_tail_buffer_{has_tail_buffer} {};
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32},
      {"m_tiles_per_wg", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t nbits_;
  SubgroupMatrixConfig config_;
  uint32_t tile_size_m_;
  uint32_t tile_size_n_;
  uint32_t tile_size_k_;
  bool has_zero_points_;
  bool has_bias_;
  bool has_weight_idx_;
  bool has_weight_idx_indirect_;
  bool has_tail_buffer_;
};

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
