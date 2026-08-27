// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Computes one inverse-RMS value per (batch, sequence, hc_mult) row. One workgroup handles one row so
// the reduction is shared by every output channel and convolution tap instead of being repeated.
class ShortConvInvRmsProgram final : public Program<ShortConvInvRmsProgram> {
 public:
  ShortConvInvRmsProgram() : Program{"ShortConvInvRms"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};

class ShortConvProgram final : public Program<ShortConvProgram> {
 public:
  ShortConvProgram(bool has_bias, bool apply_silu_or_swish)
      : Program{"ShortConv"}, has_bias_(has_bias), apply_silu_or_swish_(apply_silu_or_swish) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_size", ProgramUniformVariableDataType::Uint32},
                                          {"dilation", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool apply_silu_or_swish_;
};

class ShortConv final : public WebGpuKernel {
 public:
  explicit ShortConv(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
