// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class BitLinearQuantizeProgram final : public Program<BitLinearQuantizeProgram> {
 public:
  BitLinearQuantizeProgram(uint32_t k, uint32_t k_padded) : Program{"BitLinearQuantize"}, K_(k), K_PADDED_(k_padded) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
 private:
  uint32_t K_;
  uint32_t K_PADDED_;
};

class BitLinearMultiplyProgram final : public Program<BitLinearMultiplyProgram> {
 public:
  BitLinearMultiplyProgram() : Program{"BitLinearMultiply"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"InputAStride", ProgramUniformVariableDataType::Uint32},
                                          {"scale_B", ProgramUniformVariableDataType::Float32},
                                          {"num_N_tile", ProgramUniformVariableDataType::Uint32});
};

class BitLinearMultiplySingleMProgram final : public Program<BitLinearMultiplySingleMProgram> {
 public:
  BitLinearMultiplySingleMProgram(uint32_t tile_size_k, uint32_t tile_size) : Program{"BitLinearMultiplySingleM"},
                                                                                 tile_size_k_(tile_size_k),
                                                                                 tile_size_(tile_size) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"K20", ProgramUniformVariableDataType::Uint32},
                                          {"scale_B", ProgramUniformVariableDataType::Float32},
                                          {"num_N_tile", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_k_;
  uint32_t tile_size_;
};

class BitLinear final : public WebGpuKernel {
 public:
  BitLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    scale_b_ = info.GetAttr<float>("scale");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  float scale_b_ = 1.0f;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
