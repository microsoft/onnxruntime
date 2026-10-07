// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

#define WEBGPU_ROTARY_EMBEDDING_PROGRAM_CONFIG(F) \
  F(bool, interleaved_)                           \
  F(bool, use_seqlens_for_position_)

struct RotaryEmbeddingProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_ROTARY_EMBEDDING_PROGRAM_CONFIG);
    Config(bool interleaved, bool use_seqlens_for_position = false)
        : interleaved_{interleaved}, use_seqlens_for_position_{use_seqlens_for_position} {}
  };
  static constexpr std::string_view name = "RotaryEmbedding";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"scale", ProgramUniformVariableDataType::Float32},
                                          {"global_shape", ProgramUniformVariableDataType::Uint32},
                                          {"global_stride", ProgramUniformVariableDataType::Uint32},
                                          {"input_output_stride", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_ROTARY_EMBEDDING_PROGRAM_CONFIG

using RotaryEmbeddingProgram = ConfiguredProgram<RotaryEmbeddingProgramShader>;

#define WEBGPU_FUSED_Q_K_ROTARY_EMBEDDING_PROGRAM_CONFIG(F) \
  F(bool, interleaved_)                                     \
  F(bool, has_qk_norm_)

struct FusedQKRotaryEmbeddingProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FUSED_Q_K_ROTARY_EMBEDDING_PROGRAM_CONFIG);
    Config(bool interleaved, bool has_qk_norm) : interleaved_{interleaved}, has_qk_norm_{has_qk_norm} {}
  };
  static constexpr std::string_view name = "FusedQKRotaryEmbedding";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

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
};
#undef WEBGPU_FUSED_Q_K_ROTARY_EMBEDDING_PROGRAM_CONFIG

using FusedQKRotaryEmbeddingProgram = ConfiguredProgram<FusedQKRotaryEmbeddingProgramShader>;

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
