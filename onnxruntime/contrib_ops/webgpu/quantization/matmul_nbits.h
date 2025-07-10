// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MatMulNBitsWideTileProgram final : public Program<MatMulNBitsWideTileProgram> {
 public:
  MatMulNBitsWideTileProgram(bool has_zero_points, uint32_t tile_m, uint32_t tile_n, uint32_t nbits)
      : Program{"MatMulNBitsWideTile"}, has_zero_points_{has_zero_points}, tile_m_(tile_m), tile_n_(tile_n), nbits_(nbits) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"Batch", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"K_div_4", ProgramUniformVariableDataType::Uint32},
                                          {"K_div_32", ProgramUniformVariableDataType::Uint32},
                                          {"K_of_b", ProgramUniformVariableDataType::Uint32},
                                          {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
                                          {"num_N_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_M_tile", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_zero_points_;
  uint32_t tile_m_;
  uint32_t tile_n_;
  uint32_t nbits_;
};

class MatMulNBitsProgram final : public Program<MatMulNBitsProgram> {
 public:
  MatMulNBitsProgram(uint32_t tile_size, uint32_t nbits, bool has_zero_points) : Program{"MatMulNBits"}, tile_size_(tile_size), nbits_(nbits), has_zero_points_(has_zero_points) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_zero_points_;
};

class MatMulNBits final : public WebGpuKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8,
                "Only 4b/8b quantization is supported for MatMulNBits op, additional bits support is planned.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
