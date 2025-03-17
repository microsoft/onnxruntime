// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MatMulNBitsProgram final : public Program<MatMulNBitsProgram> {
 public:
  MatMulNBitsProgram(uint32_t output_number, uint32_t block_size, uint32_t tile_m, int components_b, bool has_zero_points, bool use_subgroup) : Program{"MatMulNBits"},
                                                                                                                                                output_number_{output_number},
                                                                                                                                                block_size_{block_size},
                                                                                                                                                tile_m_{tile_m},
                                                                                                                                                components_b_{components_b},
                                                                                                                                                has_zero_points_{has_zero_points},
                                                                                                                                                use_subgroup_(use_subgroup) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"block_size", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t output_number_;
  uint32_t block_size_;
  uint32_t tile_m_;
  int components_b_;
  bool has_zero_points_;
  bool use_subgroup_;
};

class DP4AMatMulQuantizeProgram final : public Program<DP4AMatMulQuantizeProgram> {
 public:
  DP4AMatMulQuantizeProgram() : Program{"DP4AMatMulQuantize"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_size", ProgramUniformVariableDataType::Uint32});
};

class DP4AMatMulNBitsProgram final : public Program<DP4AMatMulNBitsProgram> {
 public:
  DP4AMatMulNBitsProgram() : Program{"DP4AMatMulNBits"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K8", ProgramUniformVariableDataType::Uint32},
      {"K16", ProgramUniformVariableDataType::Uint32});
};

class MatMulNBits final : public WebGpuKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    int64_t bits = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    ORT_ENFORCE(bits == 4,
                "Only 4b quantization is supported for MatMulNBits op, additional bits support is planned.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
