// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_SPLIT_PACKED_Q_K_V_WITH_ROTARY_EMBEDDING_AND_COPY_K_V_PROGRAM_CONFIG(F) \
  F(bool, interleaved_)                                                                \
  F(bool, prepare_indirect_dispatch_)                                                  \
  F(uint32_t, multi_rotary_cache_concat_offset_)

struct SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPLIT_PACKED_Q_K_V_WITH_ROTARY_EMBEDDING_AND_COPY_K_V_PROGRAM_CONFIG);
    Config(bool interleaved, bool prepare_indirect_dispatch, uint32_t multi_rotary_cache_concat_offset)
        : interleaved_(interleaved),
          prepare_indirect_dispatch_(prepare_indirect_dispatch),
          multi_rotary_cache_concat_offset_(multi_rotary_cache_concat_offset) {}
  };
  static constexpr std::string_view name = "SplitPackedQKVWithRotaryEmbeddingAndCopyKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"half_rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"tile_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_q_tiles", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPLIT_PACKED_Q_K_V_WITH_ROTARY_EMBEDDING_AND_COPY_K_V_PROGRAM_CONFIG

using SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram =
    ConfiguredProgram<SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgramShader>;

#define WEBGPU_COPY_K_V_CACHE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                 \
  F(bool, has_past_)                            \
  F(bool, kv_BNSH_)                             \
  F(bool, past_present_share_buffer_)           \
  F(bool, prepare_indirect_dispatch_)           \
  F(bool, use_seqlen_k_)

struct CopyKVCacheProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_COPY_K_V_CACHE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool has_past, bool kv_BNSH, bool past_present_share_buffer,
           bool prepare_indirect_dispatch = false, bool use_seqlen_k = false)
        : program_name_{kernel_name},
          has_past_(has_past),
          kv_BNSH_(kv_BNSH),
          past_present_share_buffer_(past_present_share_buffer),
          prepare_indirect_dispatch_(prepare_indirect_dispatch),
          use_seqlen_k_(use_seqlen_k) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"copy_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_q_tiles", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_COPY_K_V_CACHE_PROGRAM_CONFIG

using CopyKVCacheProgram = ConfiguredProgram<CopyKVCacheProgramShader>;

#define WEBGPU_PREPARE_INDIRECT_DISPATCH_PROGRAM_CONFIG(F)

struct PrepareIndirectDispatchProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_PREPARE_INDIRECT_DISPATCH_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "PrepareIndirectDispatch";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"tile_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"num_q_tiles", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"present_sequence_length", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_PREPARE_INDIRECT_DISPATCH_PROGRAM_CONFIG

using PrepareIndirectDispatchProgram = ConfiguredProgram<PrepareIndirectDispatchProgramShader>;

#define WEBGPU_FLASH_ATTENTION_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                  \
  F(bool, has_attention_bias_)                   \
  F(bool, is_qualcomm_)                          \
  F(int, qkv_head_size_)                         \
  F(int, qkv_num_heads_)                         \
  F(bool, is_unidirectional_)                    \
  F(bool, is_nvidia_)                            \
  F(bool, use_shm_path_)                         \
  F(bool, q_BNSH_)                               \
  F(bool, use_seqlen_k_)                         \
  F(bool, has_head_sink_)                        \
  F(bool, has_local_window_)                     \
  F(int, max_k_step_)                            \
  F(bool, kv_cache_quantization_)                \
  F(uint32_t, kv_cache_quantization_bits_)       \
  F(int, compressed_head_size_u32_)              \
  F(bool, use_seqlens_q_)

struct FlashAttentionProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool has_attention_bias, bool is_qualcomm, bool is_fp16, int qkv_head_size,
           int qkv_num_heads, bool is_unidirectional, bool is_nvidia, bool is_apple, bool has_subgroups, bool q_BNSH,
           bool use_seqlen_k = false, bool has_head_sink = false, bool has_local_window = false,
           uint32_t kv_cache_quantization_bits = 0, int compressed_head_size_u32 = 0, bool use_seqlens_q = false);
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"num_seq_tile", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim3", ProgramUniformVariableDataType::Uint32},
                                          {"local_window_size", ProgramUniformVariableDataType::Uint32});

  // Per-batch new-Q-length path (LEFT-aligned Q). When set, the shader reads
  // seqlens_q[b] and computes past_sequence_length_b = total_kv_b - q_len_b.
  // When unset (default), callers keep the uniform-q_len clamp path unchanged.
};
#undef WEBGPU_FLASH_ATTENTION_PROGRAM_CONFIG

using FlashAttentionProgram = ConfiguredProgram<FlashAttentionProgramShader>;

// Dedicated single-kernel prefill program for PagedAttention. It mirrors the
// dense FlashAttentionProgram workgroup algorithm but reads K/V tiles through
// block_table instead of materializing dense K/V scratch tensors.
#define WEBGPU_FLASH_ATTENTION_PAGED_PREFILL_PROGRAM_CONFIG(F) \
  F(bool, is_fp16_)                                            \
  F(int, qkv_head_size_)                                       \
  F(int, qkv_num_heads_)                                       \
  F(bool, is_unidirectional_)                                  \
  F(int, max_k_step_)                                          \
  F(bool, q_varlen_)

struct FlashAttentionPagedPrefillProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_PAGED_PREFILL_PROGRAM_CONFIG);
    Config(bool is_fp16, int qkv_head_size, int qkv_num_heads, bool is_unidirectional, bool q_varlen)
        : is_fp16_(is_fp16),
          qkv_head_size_(qkv_head_size),
          qkv_num_heads_(qkv_num_heads),
          is_unidirectional_(is_unidirectional),
          q_varlen_(q_varlen) {
      // The shader is shared-memory only (no subgroup intrinsics). max_k_step
      // falls out of the workgroup shm budget: k_tile + v_tile = 2 *
      // element_size * head_size * max_k_step bytes.
      const int element_size = is_fp16 ? 2 : 4;
      constexpr int kMinWorkgroupStorageBudgetBytes = 16384;
      const int max_k_from_shm = kMinWorkgroupStorageBudgetBytes / (2 * element_size * qkv_head_size);
      max_k_step_ = max_k_from_shm >= 32 ? 32 : 16;
    }
  };
  static constexpr std::string_view name = "FlashAttentionPagedPrefill";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"num_seq_tile", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_FLASH_ATTENTION_PAGED_PREFILL_PROGRAM_CONFIG

using FlashAttentionPagedPrefillProgram = ConfiguredProgram<FlashAttentionPagedPrefillProgramShader>;

Status ComputeFlashAttentionPagedPrefill(onnxruntime::webgpu::ComputeContext& context,
                                         const Tensor* q,
                                         const Tensor* key_cache,
                                         const Tensor* value_cache,
                                         const Tensor* block_table,
                                         Tensor* output,
                                         const Tensor* seqlen_k,
                                         const Tensor* seqlens_q,
                                         const WebgpuAttentionParameters& parameters,
                                         uint32_t block_size,
                                         const Tensor* cumulative_seqlens_q = nullptr);

#define WEBGPU_FLASH_ATTENTION_DECODE_Q_K_V_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                               \
  F(bool, has_attention_bias_)                                \
  F(uint32_t, tile_size_)                                     \
  F(int, head_size_vec_)                                      \
  F(bool, use_indirect_dispatch_)                             \
  F(bool, q_BNSH_)                                            \
  F(bool, is_unidirectional_)                                 \
  F(uint32_t, m_tile_)                                        \
  F(bool, use_seqlen_k_)                                      \
  F(bool, kv_cache_quantization_)                             \
  F(uint32_t, kv_cache_quantization_bits_)                    \
  F(int, compressed_head_size_u32_)                           \
  F(bool, use_seqlens_q_)

struct FlashAttentionDecodeQKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_DECODE_Q_K_V_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool has_attention_bias, uint32_t tile_size, int head_size_vec,
           bool use_indirect_dispatch, bool q_BNSH = false, bool is_unidirectional = false, uint32_t m_tile = 1,
           bool use_seqlen_k = false, uint32_t kv_cache_quantization_bits = 0, int compressed_head_size_u32 = 0,
           bool use_seqlens_q = false)
        : program_name_{kernel_name},
          has_attention_bias_(has_attention_bias),
          tile_size_(tile_size),
          head_size_vec_(head_size_vec),
          use_indirect_dispatch_(use_indirect_dispatch),
          q_BNSH_(q_BNSH),
          is_unidirectional_(is_unidirectional),
          m_tile_(m_tile),
          use_seqlen_k_(use_seqlen_k),
          kv_cache_quantization_(kv_cache_quantization_bits != 0),
          kv_cache_quantization_bits_(kv_cache_quantization_bits),
          compressed_head_size_u32_(compressed_head_size_u32),
          use_seqlens_q_(use_seqlens_q) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim3", ProgramUniformVariableDataType::Uint32},
                                          {"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"local_window_size", ProgramUniformVariableDataType::Uint32});

  // See FlashAttentionProgram::use_seqlens_q_ for semantics.
};
#undef WEBGPU_FLASH_ATTENTION_DECODE_Q_K_V_PROGRAM_CONFIG

using FlashAttentionDecodeQKVProgram = ConfiguredProgram<FlashAttentionDecodeQKVProgramShader>;

#define WEBGPU_FLASH_ATTENTION_DECODE_VX_REDUCE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                                   \
  F(uint32_t, tile_size_)                                         \
  F(uint32_t, seq_tile_size_)                                     \
  F(bool, has_head_sink_)                                         \
  F(uint32_t, m_tile_)                                            \
  F(bool, use_seqlen_k_)

struct FlashAttentionDecodeVxReduceProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_DECODE_VX_REDUCE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, uint32_t tile_size, uint32_t seq_tile_size, bool has_head_sink = false,
           uint32_t m_tile = 1, bool use_seqlen_k = false)
        : program_name_{kernel_name},
          tile_size_(tile_size),
          seq_tile_size_(seq_tile_size),
          has_head_sink_(has_head_sink),
          m_tile_(m_tile),
          use_seqlen_k_(use_seqlen_k) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"num_total_seq_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_head_size_tile", ProgramUniformVariableDataType::Uint32},
                                          {"batch_heads", ProgramUniformVariableDataType::Uint32},
                                          {"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_FLASH_ATTENTION_DECODE_VX_REDUCE_PROGRAM_CONFIG

using FlashAttentionDecodeVxReduceProgram = ConfiguredProgram<FlashAttentionDecodeVxReduceProgramShader>;

// Phase 2 scaffold: keep baseline decode programs untouched by adding paged variants.
#define WEBGPU_FLASH_ATTENTION_PAGED_DECODE_Q_K_V_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                                     \
  F(bool, has_attention_bias_)                                      \
  F(uint32_t, tile_size_)                                           \
  F(int, head_size_vec_)                                            \
  F(bool, use_indirect_dispatch_)                                   \
  F(bool, q_BNSH_)                                                  \
  F(bool, is_unidirectional_)                                       \
  F(uint32_t, m_tile_)                                              \
  F(bool, use_seqlen_k_)                                            \
  F(bool, kv_cache_quantization_)                                   \
  F(uint32_t, kv_cache_quantization_bits_)                          \
  F(int, compressed_head_size_u32_)                                 \
  F(bool, use_seqlens_q_)

struct FlashAttentionPagedDecodeQKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_PAGED_DECODE_Q_K_V_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool has_attention_bias, uint32_t tile_size, int head_size_vec,
           bool use_indirect_dispatch, bool q_BNSH = false, bool is_unidirectional = false, uint32_t m_tile = 1,
           bool use_seqlen_k = false, uint32_t kv_cache_quantization_bits = 0, int compressed_head_size_u32 = 0,
           bool use_seqlens_q = false)
        : program_name_{kernel_name},
          has_attention_bias_(has_attention_bias),
          tile_size_(tile_size),
          head_size_vec_(head_size_vec),
          use_indirect_dispatch_(use_indirect_dispatch),
          q_BNSH_(q_BNSH),
          is_unidirectional_(is_unidirectional),
          m_tile_(m_tile),
          use_seqlen_k_(use_seqlen_k),
          kv_cache_quantization_(kv_cache_quantization_bits != 0),
          kv_cache_quantization_bits_(kv_cache_quantization_bits),
          compressed_head_size_u32_(compressed_head_size_u32),
          use_seqlens_q_(use_seqlens_q) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim3", ProgramUniformVariableDataType::Uint32},
                                          {"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"max_num_blocks_per_seq", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_FLASH_ATTENTION_PAGED_DECODE_Q_K_V_PROGRAM_CONFIG

using FlashAttentionPagedDecodeQKVProgram = ConfiguredProgram<FlashAttentionPagedDecodeQKVProgramShader>;

#define WEBGPU_FLASH_ATTENTION_PAGED_DECODE_VX_REDUCE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                                         \
  F(uint32_t, tile_size_)                                               \
  F(uint32_t, seq_tile_size_)                                           \
  F(bool, has_head_sink_)                                               \
  F(uint32_t, m_tile_)                                                  \
  F(bool, use_seqlen_k_)

struct FlashAttentionPagedDecodeVxReduceProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FLASH_ATTENTION_PAGED_DECODE_VX_REDUCE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, uint32_t tile_size, uint32_t seq_tile_size, bool has_head_sink = false,
           uint32_t m_tile = 1, bool use_seqlen_k = false)
        : program_name_{kernel_name},
          tile_size_(tile_size),
          seq_tile_size_(seq_tile_size),
          has_head_sink_(has_head_sink),
          m_tile_(m_tile),
          use_seqlen_k_(use_seqlen_k) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"num_total_seq_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_head_size_tile", ProgramUniformVariableDataType::Uint32},
                                          {"batch_heads", ProgramUniformVariableDataType::Uint32},
                                          {"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_FLASH_ATTENTION_PAGED_DECODE_VX_REDUCE_PROGRAM_CONFIG

using FlashAttentionPagedDecodeVxReduceProgram = ConfiguredProgram<FlashAttentionPagedDecodeVxReduceProgramShader>;

// seqlens_q (optional): int32[batch_size] of per-batch new-Q lengths. Enables
// LEFT-aligned variable-q_len callers (e.g. PagedAttention). Uniform-q_len
// callers pass nullptr and keep the pre-existing clamped path.
Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k = nullptr,
                           const Tensor* cos_cache = nullptr, const Tensor* sin_cache = nullptr, const Tensor* head_sink = nullptr,
                           const Tensor* total_seqlen = nullptr, const Tensor* seqlens_q = nullptr,
                           const Tensor* block_table = nullptr, uint32_t block_size = 0, uint32_t max_num_blocks_per_seq = 0,
                           const Tensor* cumulative_seqlens_q = nullptr, int local_window_size = -1);

// Adapter/config gate for the fused paged-prefill shader
// (FlashAttentionPagedPrefillProgram). Callers that decide up front whether Q
// arrives in packed varlen layout or padded BSNH — namely PagedAttention —
// must ask this before choosing the Q view, or the two decisions can drift and
// the shader will see a Q buffer it does not know how to index.
//
// Predicates checked (all must hold):
//   * is_fp16 — only fp16 variant is compiled today
//   * max_seqlen_q >= 32 — below that ApplyFlashAttention routes to
//     split-reduce decode instead
//   * head_size satisfies the shared-memory budget for max_k_step >= 16
//     (fp16: head_size <= 256).
//   * block_size >= max_k_step — the shader issues one block_table lookup per
//     K/V tile and reads linearly, so a tile must fit entirely inside one
//     paged block. paged_attention_helper only enforces block_size >= 16
//     (power-of-two); configs with block_size < max_k_step (e.g. block_size 16
//     and fp16 head_size <= 128 → max_k_step 32) would splice into a
//     physically-adjacent block that is not the next entry in block_table.
//
// The shader uses only shared-memory algorithms (no subgroup intrinsics), so
// it runs correctly on every WebGPU adapter — no adapter-class gate.
bool ShouldRunFusedPagedPrefill(onnxruntime::webgpu::ComputeContext& context,
                                bool is_fp16,
                                int max_seqlen_q,
                                int head_size,
                                int block_size);

bool CanApplyFlashAttention(const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);

// Split packed QKV with Q/K rotary embedding and copy KV cache fusion
Status RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV(onnxruntime::webgpu::ComputeContext& context,
                                                     const WebgpuAttentionParameters& params,
                                                     const Tensor* packedQKV,
                                                     const Tensor* seqlen_k,
                                                     const Tensor* cos_cache,
                                                     const Tensor* sin_cache,
                                                     Tensor* query,
                                                     Tensor* present_key,
                                                     Tensor* present_value,
                                                     Tensor* indirect_buffer,
                                                     uint32_t tile_size, uint32_t num_q_tiles,
                                                     const Tensor* total_seqlen = nullptr);
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
