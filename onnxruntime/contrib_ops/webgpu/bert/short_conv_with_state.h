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

// Pass 1: Compute inverse RMS per (batch, sequence, hc_mult) row and write normalized values.
class ShortConvWithStateNormProgram final : public Program<ShortConvWithStateNormProgram> {
 public:
  ShortConvWithStateNormProgram() : Program{"ShortConvWithStateNorm"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"channels", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};

// Pass 2: Dilated causal convolution with past state.
class ShortConvWithStateConvProgram final : public Program<ShortConvWithStateConvProgram> {
 public:
  ShortConvWithStateConvProgram(bool has_bias, bool has_past_state, bool apply_silu)
      : Program{"ShortConvWithStateConv"}, has_bias_(has_bias), has_past_state_(has_past_state), apply_silu_(apply_silu) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_size", ProgramUniformVariableDataType::Uint32},
                                          {"dilation", ProgramUniformVariableDataType::Uint32},
                                          {"state_len", ProgramUniformVariableDataType::Uint32},
                                          {"channels", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool has_past_state_;
  bool apply_silu_;
};

// Pass 3: Update present state from the tail of the combined timeline.
class ShortConvWithStateUpdateProgram final : public Program<ShortConvWithStateUpdateProgram> {
 public:
  ShortConvWithStateUpdateProgram(bool has_past_state)
      : Program{"ShortConvWithStateUpdate"}, has_past_state_(has_past_state) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total_state_elements", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"channels", ProgramUniformVariableDataType::Uint32},
                                          {"state_len", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_state_;
};

class ShortConvWithState final : public WebGpuKernel {
 public:
  explicit ShortConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
  int64_t kernel_size_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
