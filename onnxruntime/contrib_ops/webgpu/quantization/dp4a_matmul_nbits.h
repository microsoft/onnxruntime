// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class DP4AMatMulQuantizeProgram final : public Program<DP4AMatMulQuantizeProgram> {
 public:
  DP4AMatMulQuantizeProgram() : Program{"DP4AMatMulQuantize"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
};

class DP4AMatMulNBitsProgram final : public Program<DP4AMatMulNBitsProgram> {
 public:
  DP4AMatMulNBitsProgram(uint32_t block_size, uint32_t nbits,
                         bool has_zero_points, bool has_bias,
                         bool has_weight_idx, bool is_qualcomm) : Program{"DP4AMatMulNBits"},
                                                                  block_size_(block_size),
                                                                  nbits_(nbits),
                                                                  has_bias_(has_bias),
                                                                  has_zero_points_(has_zero_points),
                                                                  has_weight_idx_(has_weight_idx),
                                                                  is_qualcomm_(is_qualcomm) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
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

 private:
  uint32_t block_size_;
  uint32_t nbits_;
  bool has_bias_;
  bool has_zero_points_;
  bool has_weight_idx_;
  bool is_qualcomm_;
};

class DP4AMatMulNBitsSmallMProgram final : public Program<DP4AMatMulNBitsSmallMProgram> {
 public:
  DP4AMatMulNBitsSmallMProgram(uint32_t tile_size_k_vec, uint32_t tile_size, uint32_t nbits,
                               bool has_zero_points, bool has_bias,
                               bool has_weight_idx, bool single_scale_weights) : Program{"DP4AMatMulNBitsSmallMProgram"},
                                                                                 tile_size_k_vec_(tile_size_k_vec),
                                                                                 tile_size_(tile_size),
                                                                                 nbits_(nbits),
                                                                                 has_bias_(has_bias),
                                                                                 has_zero_points_(has_zero_points),
                                                                                 has_weight_idx_(has_weight_idx),
                                                                                 single_scale_weights_(single_scale_weights) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
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
      {"weight_idx", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_k_vec_;
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_bias_;
  bool has_zero_points_;
  bool has_weight_idx_;
  bool single_scale_weights_;
};

Status ApplyDP4AMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points, const Tensor* bias,
                                  uint32_t batch_count,
                                  uint32_t M,
                                  uint32_t N,
                                  uint32_t K,
                                  uint32_t block_size,
                                  uint32_t zero_blocks_per_col,
                                  uint32_t min_M_for_tile_optimization,
                                  uint32_t nbits,
                                  onnxruntime::webgpu::ComputeContext& context,
                                  Tensor* y,
                                  const uint32_t weight_index);

bool CanApplyDP4AMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                   uint64_t accuracy_level,
                                   uint32_t block_size,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t components_k);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
