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
};

class DP4AMatMulNBitsProgram final : public Program<DP4AMatMulNBitsProgram> {
 public:
  DP4AMatMulNBitsProgram(uint32_t block_size, uint32_t nbits) : Program{"DP4AMatMulNBits"}, block_size_(block_size), nbits_(nbits) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K8", ProgramUniformVariableDataType::Uint32},
      {"K16", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t block_size_;
  uint32_t nbits_;
};

class DP4AMatMulNBitsSmallMProgram final : public Program<DP4AMatMulNBitsSmallMProgram> {
 public:
  DP4AMatMulNBitsSmallMProgram(uint32_t tile_size_k_vec, uint32_t tile_size, uint32_t nbits) : Program{"DP4AMatMulNBitsSmallMProgram"}, tile_size_k_vec_(tile_size_k_vec), tile_size_(tile_size), nbits_(nbits) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K16", ProgramUniformVariableDataType::Uint32},
      {"K32", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_k_vec_;
  uint32_t tile_size_;
  uint32_t nbits_;
};

Status ApplyDP4AMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  uint32_t M,
                                  uint32_t N,
                                  uint32_t K,
                                  uint32_t block_size,
                                  uint32_t min_M_for_tile_optimization,
                                  uint32_t nbits,
                                  onnxruntime::webgpu::ComputeContext& context,
                                  Tensor* y);

bool CanApplyDP4AMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                   uint64_t accuracy_level,
                                   uint32_t block_size,
                                   uint32_t batch_count,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t components_k,
                                   bool has_zero_points);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
