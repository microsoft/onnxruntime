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

class RotaryEmbeddingProgram final : public Program<RotaryEmbeddingProgram> {
 public:
  RotaryEmbeddingProgram(bool interleaved) : Program{"RotaryEmbedding"}, interleaved_{interleaved} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"scale", ProgramUniformVariableDataType::Float32},
                                          {"global_shape", ProgramUniformVariableDataType::Uint32},
                                          {"global_stride", ProgramUniformVariableDataType::Uint32},
                                          {"input_output_stride", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
};

class FusedQKRotaryEmbeddingProgram final : public Program<FusedQKRotaryEmbeddingProgram> {
 public:
  FusedQKRotaryEmbeddingProgram(bool interleaved, bool has_qk_norm)
      : Program{"FusedQKRotaryEmbedding"},
        interleaved_{interleaved},
        has_qk_norm_{has_qk_norm} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // q_* describes query rotation domain (same definition as existing program)
  // k_* describes key rotation domain.
  // When has_qk_norm_ is true, the program also fuses a per-head RMS normalization
  // (epsilon = qk_norm_epsilon, scale = q_norm_weight / k_norm_weight) over the
  // head_size channels of Q and K before the rotary rotation. head_size and
  // qk_norm_epsilon are required uniforms when has_qk_norm_ is true; they are
  // ignored otherwise but must still be supplied (callers pass placeholder values).
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"scale", ProgramUniformVariableDataType::Float32},
      {"q_global_shape", ProgramUniformVariableDataType::Uint32},
      {"q_global_stride", ProgramUniformVariableDataType::Uint32},
      {"q_input_output_stride", ProgramUniformVariableDataType::Uint32},
      {"k_global_shape", ProgramUniformVariableDataType::Uint32},
      {"k_input_output_stride", ProgramUniformVariableDataType::Uint32},
      {"q_domain_size", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"qk_norm_epsilon", ProgramUniformVariableDataType::Float32});

 private:
  const bool interleaved_;
  const bool has_qk_norm_;
};

class RotaryEmbedding final : public WebGpuKernel {
 public:
  RotaryEmbedding(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float scale_;
  int num_heads_;
  int rotary_embedding_dim_;
  bool interleaved_;
  bool is_packed_batching_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
