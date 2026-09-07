// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// Scatter unpacked K and V into the paged (block-based) KV cache.
// Plain non-fused scatter (no rotary, no packing).
//
// Inputs (all read):
//   key                          : (token_count, kv_hidden_size)      [T]
//   value                        : (token_count, kv_hidden_size)      [T]
//   cumulative_sequence_length_q : (batch_size + 1,)                  [S]
//   past_seqlens                 : (batch_size,)                      [S]
//   block_table                  : (batch_size, max_num_blocks_per_seq) [S]
//
// Outputs (write, aliased by the calling op with the corresponding cache inputs):
//   key_cache   : (num_blocks, block_size, kv_num_heads, head_size)   [T]
//   value_cache : (num_blocks, block_size, kv_num_heads, head_size)   [T]
//
// Dispatch model: one invocation per (token_idx, kv_head_idx, dim_idx),
// unrolled row-major into a single 1-D dispatch. Each invocation writes one
// element into both caches (see the .wgsl.template for the address model).
class ScatterKVToPagedCacheProgram final : public Program<ScatterKVToPagedCacheProgram> {
 public:
  ScatterKVToPagedCacheProgram() : Program{"ScatterKVToPagedCache"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"token_count", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Rotary embedding (RoPE) for one packed 2-D tensor of the shape
//   (token_count, n_heads * head_size)
// used by the WebGPU PagedAttention op for both the query and key rotation
// stages. Value is not rotated.
//
// Inputs (all read):
//   input                        : (token_count, n_heads * head_size)  [T]
//   cos_cache                    : (M, rotary_dim / 2)                 [T]
//   sin_cache                    : (M, rotary_dim / 2)                 [T]
//   cumulative_sequence_length_q : (batch_size + 1,)                   [S]
//   past_seqlens                 : (batch_size,)                       [S]
//
// Output (write):
//   output                       : (token_count, n_heads * head_size)  [T]
//
// The math matches paged_attention_impl.cu::RotaryEmbeddingTNH (`interleaved`
// vs. split layout), keyed off `rotary_interleaved` propagated as a uniform.
// `n_heads` is `num_heads` for the query rotation and `kv_num_heads` for the
// key rotation; the same program handles both by taking `n_heads` as a
// uniform. Dims `>= rotary_dim` are copied through unchanged (matches CUDA).
class PagedAttentionRotaryProgram final : public Program<PagedAttentionRotaryProgram> {
 public:
  PagedAttentionRotaryProgram() : Program{"PagedAttentionRotary"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"n_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"interleaved", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Split a packed-QKV tensor into three separate Q, K, V tensors so the rest
// of the WebGPU PagedAttention pipeline can consume them the same way it
// consumes the non-packed layout.
//
// Input row layout (row-major within each token):
//   cols [0, q_hidden_size)                            → Q
//   cols [q_hidden_size, q_hidden_size + kv_hidden)    → K
//   cols [q_hidden_size + kv_hidden, packed_hidden)    → V
//
// Inputs:
//   input : (token_count, q_hidden_size + 2 * kv_hidden_size)  [T]
//
// Outputs:
//   q_out : (token_count, q_hidden_size)                       [T]
//   k_out : (token_count, kv_hidden_size)                      [T]
//   v_out : (token_count, kv_hidden_size)                      [T]
class PagedAttentionSplitPackedQKVProgram final
    : public Program<PagedAttentionSplitPackedQKVProgram> {
 public:
  PagedAttentionSplitPackedQKVProgram() : Program{"PagedAttentionSplitPackedQKV"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"token_count", ProgramUniformVariableDataType::Uint32},
      {"q_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"packed_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Gather paged K/V into padded contiguous BNSH scratch tensors so
// ApplyFlashAttention can consume them the same way GQA/Attention consume
// their present_key/present_value tensors.
//
// Inputs (all read):
//   key_cache   : (num_blocks, block_size, kv_num_heads, head_size)      [T]
//   value_cache : (num_blocks, block_size, kv_num_heads, head_size)      [T]
//   cumulative_sequence_length : (batch_size + 1,)                       [S]
//   past_seqlens               : (batch_size,)                           [S]
//   block_table                : (batch_size, max_num_blocks_per_seq)    [S]
//
// Outputs (write):
//   k_padded : (batch_size, kv_num_heads, max_kv_len, head_size)         [T]  BNSH
//   v_padded : (batch_size, kv_num_heads, max_kv_len, head_size)         [T]  BNSH
//
// Slots [s >= total_kv_len_b) are zero-filled so FlashAttention's dot-product
// contributions are zero there. (Combined with the per-batch causal mask driven
// by seqlen_k, the pad tokens produce no leakage into the visible logits.)
class PagedAttentionGatherKVProgram final : public Program<PagedAttentionGatherKVProgram> {
 public:
  PagedAttentionGatherKVProgram() : Program{"PagedAttentionGatherKV"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"max_kv_len", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Unpack packed varlen Q into padded BSNH so it can be fed to
// ApplyFlashAttention (which expects a real batch dimension).
//
// Inputs (all read):
//   input                        : (token_count, num_heads * head_size)  [T]  packed
//   cumulative_sequence_length   : (batch_size + 1,)                     [S]
//
// Output (write):
//   output : (batch_size, max_seqlen_q, num_heads, head_size)            [T]  BSNH
//
// Pad slots (s >= seq_len_b) hold zero; their outputs from FlashAttention
// are discarded by the repack kernel and never surface in the packed
// PagedAttention output tensor.
class PagedAttentionUnpackQueryProgram final : public Program<PagedAttentionUnpackQueryProgram> {
 public:
  PagedAttentionUnpackQueryProgram() : Program{"PagedAttentionUnpackQuery"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"max_seqlen_q", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Repack padded BSNH FlashAttention output back to the packed varlen layout
// PagedAttention's caller expects. Inverse of PagedAttentionUnpackQuery.
//
// Inputs (read):
//   input                        : (batch_size, max_seqlen_q, num_heads, head_size)  [T]
//   cumulative_sequence_length   : (batch_size + 1,)                                 [S]
//
// Output (write):
//   output : (token_count, num_heads * head_size)                                    [T]
class PagedAttentionRepackOutputProgram final : public Program<PagedAttentionRepackOutputProgram> {
 public:
  PagedAttentionRepackOutputProgram() : Program{"PagedAttentionRepackOutput"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Pack the two device-resident metadata tensors into one contiguous buffer so
// PagedAttention can perform one host readback rather than two. The layout is
// [cumulative_sequence_length[0..batch_size], past_seqlens[0..batch_size)).
class PagedAttentionPackMetadataProgram final : public Program<PagedAttentionPackMetadataProgram> {
 public:
  PagedAttentionPackMetadataProgram() : Program{"PagedAttentionPackMetadata"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Op contract, phased delivery plan, and reuse strategy are documented in
// docs/design/webgpu_paged_attention.md.
class PagedAttention final : public WebGpuKernel {
 public:
  explicit PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  float softcap_;
  float qk_norm_epsilon_;
  std::string k_quant_type_;
  std::string v_quant_type_;
  std::string k_cache_dtype_;
  std::string v_cache_dtype_;
  std::string kv_cache_layout_;
  int v_head_size_;
  int rotary_offset_;
  bool use_smooth_softmax_;
  bool has_explicit_scale_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
