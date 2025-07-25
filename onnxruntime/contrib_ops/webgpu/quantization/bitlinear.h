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
  BitLinearQuantizeProgram(uint32_t k) : Program{"BitLinearQuantize"}, K_(k) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
 private:
  uint32_t K_;
};

class BitLinearMultiplyProgram final : public Program<BitLinearMultiplyProgram> {
 public:
  BitLinearMultiplyProgram() : Program{"BitLinearMultiply"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"InputAStride", ProgramUniformVariableDataType::Uint32},
                                          {"scale_B", ProgramUniformVariableDataType::Float32});
};

class BitLinear final : public WebGpuKernel {
 public:
  BitLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    scale_b_ = info.GetAttr<float>("scale");

    // Validate that K is divisible by 5 for ternary packing
    ORT_ENFORCE(K_ % 5 == 0, "K must be divisible by 5 for BitLinear ternary packing, got K=", K_);
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
