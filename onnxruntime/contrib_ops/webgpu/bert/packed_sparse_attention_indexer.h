// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// Copies past_* state into present_* state unchanged (element-for-element); used as the baseline
// before the update programs below overwrite only the newly produced entries. Works for any
// tensor element type via UseElementTypeAlias, so it is reused for key_state / kv_buffer /
// gate_buffer (T) and state_lengths (int32).
#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_COPY_PROGRAM_CONFIG(F)

struct PackedSparseAttentionIndexerCopyProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_COPY_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"dst_offset", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_COPY_PROGRAM_CONFIG

using PackedSparseAttentionIndexerCopyProgram = ConfiguredProgram<PackedSparseAttentionIndexerCopyProgramShader>;

// One invocation per request: forms every newly-closed compress_ratio block (mean-pool ->
// RMSNorm -> leading RoPE -> append) and publishes the raw trailing buffer.
#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_UPDATE_PROGRAM_CONFIG(F) \
  F(bool, cos_cache_batched_)                                               \
  F(bool, capture_state_update_)                                            \
  F(bool, has_state_update_active_)

struct PackedSparseAttentionIndexerQsaUpdateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_UPDATE_PROGRAM_CONFIG);
    Config(bool cos_cache_batched, bool capture_state_update, bool has_state_update_active)
        : cos_cache_batched_{cos_cache_batched},
          capture_state_update_{capture_state_update},
          has_state_update_active_{has_state_update_active} {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerQsaUpdate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"key_row_stride", ProgramUniformVariableDataType::Uint32},
      {"key_offset", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"buffer_capacity", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"state_update_capacity", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_UPDATE_PROGRAM_CONFIG

using PackedSparseAttentionIndexerQsaUpdateProgram =
    ConfiguredProgram<PackedSparseAttentionIndexerQsaUpdateProgramShader>;

// One invocation per query token: rotates the query, scores it against every causally visible
// prepared key_state entry, selects the token_budget / compress_ratio highest scoring blocks, and
// appends the causally visible tokens of the trailing incomplete block.
#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG(F) F(bool, cos_cache_batched_)

struct PackedSparseAttentionIndexerQsaSelectProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG);
    Config(bool cos_cache_batched) : cos_cache_batched_{cos_cache_batched} {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerQsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_row_stride", ProgramUniformVariableDataType::Uint32},
      {"query_norm_offset", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG

using PackedSparseAttentionIndexerQsaSelectProgram =
    ConfiguredProgram<PackedSparseAttentionIndexerQsaSelectProgramShader>;

// One invocation per request: closes every new compression window (softmax-gated pool -> RMSNorm
// -> trailing RoPE -> append) and publishes the raw overlap+leftover buffer.
#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_UPDATE_PROGRAM_CONFIG(F) F(bool, cos_cache_batched_)

struct PackedSparseAttentionIndexerCsaUpdateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_UPDATE_PROGRAM_CONFIG);
    Config(bool cos_cache_batched) : cos_cache_batched_{cos_cache_batched} {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerCsaUpdate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"buffer_capacity", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_UPDATE_PROGRAM_CONFIG

using PackedSparseAttentionIndexerCsaUpdateProgram =
    ConfiguredProgram<PackedSparseAttentionIndexerCsaUpdateProgramShader>;

#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_CAPTURE_PROGRAM_CONFIG(F) F(bool, has_state_update_active_)

struct PackedSparseAttentionIndexerQsaCaptureProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_CAPTURE_PROGRAM_CONFIG);
    Config(bool has_state_update_active) : has_state_update_active_{has_state_update_active} {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerQsaCapture";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"key_row_stride", ProgramUniformVariableDataType::Uint32},
      {"key_offset", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"state_update_capacity", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_QSA_CAPTURE_PROGRAM_CONFIG

using PackedSparseAttentionIndexerQsaCaptureProgram =
    ConfiguredProgram<PackedSparseAttentionIndexerQsaCaptureProgramShader>;

// One invocation per query token: rotates the query, scores it against every causally visible
// compressed key_state entry, and selects the index_topk highest scoring entries.
#define WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG(F) F(bool, cos_cache_batched_)

struct PackedSparseAttentionIndexerCsaSelectProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG);
    Config(bool cos_cache_batched) : cos_cache_batched_{cos_cache_batched} {}
  };
  static constexpr std::string_view name = "PackedSparseAttentionIndexerCsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"buffer_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"index_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"head_weight_scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_PACKED_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG

using PackedSparseAttentionIndexerCsaSelectProgram =
    ConfiguredProgram<PackedSparseAttentionIndexerCsaSelectProgramShader>;

class PackedSparseAttentionIndexer final : public WebGpuKernel {
 public:
  explicit PackedSparseAttentionIndexer(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  Status ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const;
  Status ComputeCsa(onnxruntime::webgpu::ComputeContext& context) const;

  packed_sparse_attention_indexer::Policy policy_;
  int64_t compress_ratio_;
  int64_t state_capacity_;
  int64_t state_update_capacity_;
  int64_t token_budget_;
  int64_t index_topk_;
  float epsilon_;
  float scale_;
  float head_weight_scale_;
  bool has_scale_;
  bool has_head_weight_scale_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
