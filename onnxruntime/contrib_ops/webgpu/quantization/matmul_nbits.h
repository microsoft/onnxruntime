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
  MatMulNBitsWideTileProgram(bool has_zero_points, uint32_t tile_m, uint32_t tile_n, uint32_t nbits, uint32_t weight_index)
      : Program{"MatMulNBitsWideTile"}, has_zero_points_{has_zero_points}, tile_m_(tile_m), tile_n_(tile_n), nbits_(nbits), weight_index_(weight_index) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"Batch", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K_of_a", ProgramUniformVariableDataType::Uint32},
                                          {"K_of_b", ProgramUniformVariableDataType::Uint32},
                                          {"n_blocks_per_col", ProgramUniformVariableDataType::Uint32},
                                          {"zero_blocks_per_col", ProgramUniformVariableDataType::Uint32},
                                          {"num_N_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_M_tile", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_zero_points_;
  uint32_t tile_m_;
  uint32_t tile_n_;
  uint32_t nbits_;
  uint32_t weight_index_;
};

class MatMulNBitsProgram final : public Program<MatMulNBitsProgram> {
 public:
  MatMulNBitsProgram(uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias, uint32_t weight_index, bool single_scale_weights) : Program{"MatMulNBits"}, tile_size_(tile_size), nbits_(nbits), has_zero_points_(has_zero_points), has_bias_(has_bias), weight_index_(weight_index), single_scale_weights_(single_scale_weights) {}
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
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_idx", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t nbits_;
  bool has_zero_points_;
  bool has_bias_;
  uint32_t weight_index_;
  bool single_scale_weights_;
};

class MatMulNBits final : public WebGpuKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8 || bits_ == 2,
                "Only 4b/8b/2b quantization is supported for MatMulNBits op, additional bits support is planned.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
};

Status ApplyMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level, int64_t bits_op,
                        onnxruntime::webgpu::ComputeContext& context, Tensor* y, const Tensor* offsets);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
