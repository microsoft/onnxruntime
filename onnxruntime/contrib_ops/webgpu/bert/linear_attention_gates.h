// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/bert/linear_attention_gates_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// decay = decay_scale * Softplus(a + dt_bias), beta = Sigmoid(b).
class LinearAttentionGateProgram final : public Program<LinearAttentionGateProgram> {
 public:
  LinearAttentionGateProgram(bool has_b, bool has_beta) : Program{"LinearAttentionGate"}, has_b_(has_b), has_beta_(has_beta) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_b_;
  bool has_beta_;
};

class LinearAttentionGate final : public WebGpuKernel {
 public:
  LinearAttentionGate(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(ComputeContext& context) const override;
};

// Y = X * rsqrt(mean(X^2) + epsilon) * scale * gate_activation(gate).
class GatedRMSNormProgram final : public Program<GatedRMSNormProgram> {
 public:
  GatedRMSNormProgram(GatedRMSNormActivation activation) : Program{"GatedRMSNorm"}, activation_(activation) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"norm_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  GatedRMSNormActivation activation_;
};

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
