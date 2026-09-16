// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// Copies past_* state into present_* state unchanged (element-for-element); used as the baseline
// before the update programs below overwrite only the newly produced entries. Works for any
// tensor element type via UseElementTypeAlias, so it is reused for key_state / kv_buffer /
// gate_buffer (T) and state_lengths (int32).
class PackedSparseAttentionIndexerCopyProgram final
    : public Program<PackedSparseAttentionIndexerCopyProgram> {
 public:
  PackedSparseAttentionIndexerCopyProgram() : Program{"PackedSparseAttentionIndexerCopy"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32});
};

// One invocation per request: forms every newly-closed compress_ratio block (mean-pool ->
// RMSNorm -> leading RoPE -> append) and publishes the raw trailing buffer.
class PackedSparseAttentionIndexerQsaUpdateProgram final
    : public Program<PackedSparseAttentionIndexerQsaUpdateProgram> {
 public:
  PackedSparseAttentionIndexerQsaUpdateProgram(bool cos_cache_batched, bool kv_buffer_aliases,
                                               bool state_lengths_aliases)
      : Program{"PackedSparseAttentionIndexerQsaUpdate"},
        cos_cache_batched_{cos_cache_batched},
        kv_buffer_aliases_{kv_buffer_aliases},
        state_lengths_aliases_{state_lengths_aliases} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
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

 private:
  bool cos_cache_batched_;
  bool kv_buffer_aliases_;
  bool state_lengths_aliases_;
};

// One invocation per query token: rotates the query, scores it against every causally visible
// prepared key_state entry, selects the token_budget / compress_ratio highest scoring blocks, and
// appends the causally visible tokens of the trailing incomplete block.
class PackedSparseAttentionIndexerQsaSelectProgram final
    : public Program<PackedSparseAttentionIndexerQsaSelectProgram> {
 public:
  PackedSparseAttentionIndexerQsaSelectProgram(bool cos_cache_batched, bool has_position_ids)
      : Program{"PackedSparseAttentionIndexerQsaSelect"},
        cos_cache_batched_{cos_cache_batched},
        has_position_ids_{has_position_ids} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  bool cos_cache_batched_;
  bool has_position_ids_;
};

// One invocation per request: closes every new compression window (softmax-gated pool -> RMSNorm
// -> trailing RoPE -> append) and publishes the raw overlap+leftover buffer.
class PackedSparseAttentionIndexerCsaUpdateProgram final
    : public Program<PackedSparseAttentionIndexerCsaUpdateProgram> {
 public:
  PackedSparseAttentionIndexerCsaUpdateProgram(bool cos_cache_batched, bool kv_buffer_aliases,
                                               bool gate_buffer_aliases, bool state_lengths_aliases)
      : Program{"PackedSparseAttentionIndexerCsaUpdate"},
        cos_cache_batched_{cos_cache_batched},
        kv_buffer_aliases_{kv_buffer_aliases},
        gate_buffer_aliases_{gate_buffer_aliases},
        state_lengths_aliases_{state_lengths_aliases} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
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

 private:
  bool cos_cache_batched_;
  bool kv_buffer_aliases_;
  bool gate_buffer_aliases_;
  bool state_lengths_aliases_;
};

// One invocation per query token: rotates the query, scores it against every causally visible
// compressed key_state entry, and selects the index_topk highest scoring entries.
class PackedSparseAttentionIndexerCsaSelectProgram final
    : public Program<PackedSparseAttentionIndexerCsaSelectProgram> {
 public:
  explicit PackedSparseAttentionIndexerCsaSelectProgram(bool cos_cache_batched)
      : Program{"PackedSparseAttentionIndexerCsaSelect"}, cos_cache_batched_{cos_cache_batched} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"state_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"index_topk", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"head_weight_scale", ProgramUniformVariableDataType::Float32});

 private:
  bool cos_cache_batched_;
};

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
