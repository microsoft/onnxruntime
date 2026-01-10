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
  FusedQKRotaryEmbeddingProgram(bool interleaved) : Program{"FusedQKRotaryEmbedding"}, interleaved_{interleaved} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // q_* describes query rotation domain (same definition as existing program)
  // k_* describes key rotation domain
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"scale", ProgramUniformVariableDataType::Float32},
      {"q_global_shape", ProgramUniformVariableDataType::Uint32},
      {"q_global_stride", ProgramUniformVariableDataType::Uint32},
      {"q_input_output_stride", ProgramUniformVariableDataType::Uint32},
      {"k_global_shape", ProgramUniformVariableDataType::Uint32},
      {"k_input_output_stride", ProgramUniformVariableDataType::Uint32},
      {"q_domain_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
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
