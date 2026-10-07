// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/linear_attention_gates_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::contrib::linear_attention_gates_helper;
using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// decay = decay_scale * Softplus(a + dt_bias), beta = Sigmoid(b).
#define WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_CONFIG(F) \
  F(bool, has_b_)                                      \
  F(bool, has_beta_)

struct LinearAttentionGateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_CONFIG);
    Config(bool has_b, bool has_beta) : has_b_(has_b), has_beta_(has_beta) {}
  };
  static constexpr std::string_view name = "LinearAttentionGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_CONFIG

using LinearAttentionGateProgram = ConfiguredProgram<LinearAttentionGateProgramShader>;

class LinearAttentionGate final : public WebGpuKernel {
 public:
  LinearAttentionGate(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(ComputeContext& context) const override;
};

// Y = X * rsqrt(mean(X^2) + epsilon) * scale * gate_activation(gate).
#define WEBGPU_GATED_R_M_S_NORM_PROGRAM_CONFIG(F) F(GatedRMSNormActivation, activation_)

struct GatedRMSNormProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_R_M_S_NORM_PROGRAM_CONFIG);
    Config(GatedRMSNormActivation activation) : activation_(activation) {}
  };
  static constexpr std::string_view name = "GatedRMSNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"norm_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_GATED_R_M_S_NORM_PROGRAM_CONFIG

using GatedRMSNormProgram = ConfiguredProgram<GatedRMSNormProgramShader>;

class GatedRMSNorm final : public WebGpuKernel {
 public:
  GatedRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  GatedRMSNormActivation activation_;
  float epsilon_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
