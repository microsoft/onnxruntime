// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
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
#define WEBGPU_SCATTER_K_V_TO_PAGED_CACHE_PROGRAM_CONFIG(F)

struct ScatterKVToPagedCacheProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SCATTER_K_V_TO_PAGED_CACHE_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "ScatterKVToPagedCache";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"token_count", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"num_blocks", ProgramUniformVariableDataType::Uint32},
      {"max_num_blocks_per_seq", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SCATTER_K_V_TO_PAGED_CACHE_PROGRAM_CONFIG

using ScatterKVToPagedCacheProgram = ConfiguredProgram<ScatterKVToPagedCacheProgramShader>;

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
#define WEBGPU_PAGED_ATTENTION_ROTARY_PROGRAM_CONFIG(F)

struct PagedAttentionRotaryProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_ROTARY_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionRotary";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"n_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"interleaved", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_ROTARY_PROGRAM_CONFIG

using PagedAttentionRotaryProgram = ConfiguredProgram<PagedAttentionRotaryProgramShader>;

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
#define WEBGPU_PAGED_ATTENTION_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG(F)

struct PagedAttentionSplitPackedQKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionSplitPackedQKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"token_count", ProgramUniformVariableDataType::Uint32},
      {"q_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"packed_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG

using PagedAttentionSplitPackedQKVProgram = ConfiguredProgram<PagedAttentionSplitPackedQKVProgramShader>;

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
#define WEBGPU_PAGED_ATTENTION_GATHER_K_V_PROGRAM_CONFIG(F)

struct PagedAttentionGatherKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_GATHER_K_V_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionGatherKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"max_kv_len", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_GATHER_K_V_PROGRAM_CONFIG

using PagedAttentionGatherKVProgram = ConfiguredProgram<PagedAttentionGatherKVProgramShader>;

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
#define WEBGPU_PAGED_ATTENTION_UNPACK_QUERY_PROGRAM_CONFIG(F)

struct PagedAttentionUnpackQueryProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_UNPACK_QUERY_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionUnpackQuery";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"max_seqlen_q", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_UNPACK_QUERY_PROGRAM_CONFIG

using PagedAttentionUnpackQueryProgram = ConfiguredProgram<PagedAttentionUnpackQueryProgramShader>;

// Repack padded BSNH FlashAttention output back to the packed varlen layout
// PagedAttention's caller expects. Inverse of PagedAttentionUnpackQuery.
//
// Inputs (read):
//   input                        : (batch_size, max_seqlen_q, num_heads, head_size)  [T]
//   cumulative_sequence_length   : (batch_size + 1,)                                 [S]
//
// Output (write):
//   output : (token_count, num_heads * head_size)                                    [T]
#define WEBGPU_PAGED_ATTENTION_REPACK_OUTPUT_PROGRAM_CONFIG(F)

struct PagedAttentionRepackOutputProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_REPACK_OUTPUT_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionRepackOutput";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_REPACK_OUTPUT_PROGRAM_CONFIG

using PagedAttentionRepackOutputProgram = ConfiguredProgram<PagedAttentionRepackOutputProgramShader>;

// Pack the two device-resident metadata tensors into one contiguous buffer so
// PagedAttention can perform one host readback rather than two. The layout is
// [cumulative_sequence_length[0..batch_size], past_seqlens[0..batch_size)).
#define WEBGPU_PAGED_ATTENTION_PACK_METADATA_PROGRAM_CONFIG(F)

struct PagedAttentionPackMetadataProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_PACK_METADATA_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionPackMetadata";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_PACK_METADATA_PROGRAM_CONFIG

using PagedAttentionPackMetadataProgram = ConfiguredProgram<PagedAttentionPackMetadataProgramShader>;

// Derive the exact per-request query and KV lengths on GPU when the caller
// supplies host-side upper bounds through attention_metadata.
#define WEBGPU_PAGED_ATTENTION_PREPARE_METADATA_PROGRAM_CONFIG(F)

struct PagedAttentionPrepareMetadataProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PAGED_ATTENTION_PREPARE_METADATA_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PagedAttentionPrepareMetadata";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PAGED_ATTENTION_PREPARE_METADATA_PROGRAM_CONFIG

using PagedAttentionPrepareMetadataProgram = ConfiguredProgram<PagedAttentionPrepareMetadataProgramShader>;

// Dispatch helpers shared with SparsePagedAttention, which reuses the
// prologue programs above verbatim (packed-QKV split, rotary, and the
// block_table-driven cache scatter).
Status RunPagedAttentionScatterKVToPagedCache(onnxruntime::webgpu::ComputeContext& context,
                                              const PagedAttentionParameters& parameters,
                                              const Tensor* key,
                                              const Tensor* value,
                                              const Tensor* cumulative_seqlens_q,
                                              const Tensor* past_seqlens,
                                              const Tensor* block_table,
                                              Tensor* key_cache_out,
                                              Tensor* value_cache_out);

Status RunPagedAttentionRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                        const PagedAttentionParameters& parameters,
                                        uint32_t n_heads,
                                        bool interleaved,
                                        const Tensor* input,
                                        const Tensor* cos_cache,
                                        const Tensor* sin_cache,
                                        const Tensor* cumulative_seqlens_q,
                                        const Tensor* past_seqlens,
                                        Tensor* output);

Status RunPagedAttentionSplitPackedQKV(onnxruntime::webgpu::ComputeContext& context,
                                       const PagedAttentionParameters& parameters,
                                       const Tensor* packed_qkv,
                                       Tensor* q_out,
                                       Tensor* k_out,
                                       Tensor* v_out);

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
  bool is_causal_;
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
