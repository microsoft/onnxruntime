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

class MatMulNBitsWideTileProgram final : public Program<MatMulNBitsWideTileProgram> {
 public:
  MatMulNBitsWideTileProgram(bool has_zero_points, uint32_t tile_m, uint32_t tile_n)
      : Program{"MatMulNBitsWideTileProgram"}, has_zero_points_{has_zero_points}, tile_m_(tile_m), tile_n_(tile_n) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"block_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_zero_points_;
  uint32_t tile_m_;
  uint32_t tile_n_;
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
