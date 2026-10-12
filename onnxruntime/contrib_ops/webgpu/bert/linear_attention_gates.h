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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_b, bool has_beta) {
    Config result{};
    result.has_b_ = bool(has_b);
    result.has_beta_ = bool(has_beta);

    return result;
  }
  static constexpr std::string_view name = "LinearAttentionGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                        \
  U(Uint32, num_heads)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LINEAR_ATTENTION_GATE_PROGRAM_SHADER_UNIFORMS
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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_R_M_S_NORM_PROGRAM_CONFIG);
  static Config MakeConfig(GatedRMSNormActivation activation) {
    Config result{};
    result.activation_ = GatedRMSNormActivation(activation);

    return result;
  }
  static constexpr std::string_view name = "GatedRMSNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_GATED_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, norm_size)                                     \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS
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
