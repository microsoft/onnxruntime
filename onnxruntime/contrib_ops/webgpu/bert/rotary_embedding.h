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
  RotaryEmbeddingProgram(bool interleaved, bool use_seqlens_for_position = false)
      : Program{"RotaryEmbedding"}, interleaved_{interleaved}, use_seqlens_for_position_{use_seqlens_for_position} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"scale", ProgramUniformVariableDataType::Float32},
                                          {"global_shape", ProgramUniformVariableDataType::Uint32},
                                          {"global_stride", ProgramUniformVariableDataType::Uint32},
                                          {"input_output_stride", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
  const bool use_seqlens_for_position_;
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

// Apply rotary embedding to a single tensor using RotaryEmbeddingProgram.
//
// If use_seqlens_for_position is true, `position_ids_or_seqlens` must be the seqlens tensor (shape
// [batch_size], containing per-batch seqlen_k values where
// seqlen_k = past_sequence_length + kv_sequence_length - 1). The shader derives position_id
// per batch as: past_seqlen + sequence_index, where
// past_seqlen = (seqlens[batch] + 1) - global_shape[1].
//
// If use_seqlens_for_position is false, `position_ids_or_seqlens` must be the position_ids tensor
// (shape [batch, seq] or [1, 1] for broadcast). The shader reads position from this tensor
// directly.
Status RunRotaryEmbedding(ComputeContext& context,
                          const Tensor* input,
                          const Tensor* position_ids_or_seqlens,
                          const Tensor* cos_cache,
                          const Tensor* sin_cache,
                          Tensor* output,
                          int batch_size,
                          int sequence_length,
                          int hidden_size,
                          int head_size,
                          float scale,
                          bool rotary_interleaved,
                          bool use_seqlens_for_position,
                          const std::vector<uint32_t>& input_output_strides);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
