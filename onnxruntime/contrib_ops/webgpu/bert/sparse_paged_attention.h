// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ConfiguredProgram;
using onnxruntime::webgpu::ConfiguredShaderHelper;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::WebGpuKernel;

// 'attention_mode' attribute.
enum class SparseAttentionMode {
  kSelectedOnly,
  kLocalPlusSelected,
};

// 'selected_kv_source' attribute.
enum class SparseSelectedKvSource {
  kMain,
  kAuxiliary,
};

// Scatter the current K and V into the paged main cache using the
// scheduler-provided slot_mapping instead of block_table addressing.
//
// Inputs (all read):
//   key          : (token_count, kv_hidden_size)                    [T]
//   value        : (token_count, kv_hidden_size)                    [T]
//   slot_mapping : (token_count,)                                   [S]
//
// Outputs (write, aliased by the calling op with the cache inputs):
//   key_cache   : (num_blocks, block_size, kv_num_heads, head_size) [T]
//   value_cache : (num_blocks, block_size, kv_num_heads, head_size) [T]
//
// slot_mapping holds a flat slot index (physical_block * block_size +
// slot_in_block). A negative entry suppresses the write for that token, which
// is how schedulers express "this token is already resident" or "drop it".
// Entries outside the cache are also suppressed; the value is validated as i32
// before it is ever converted to u32.
#define WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_CONFIG(F)

struct SparsePagedAttentionScatterKVProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "SparsePagedAttentionScatterKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, kv_num_heads)                                                    \
  U(Uint32, head_size)                                                       \
  U(Uint32, cache_slot_count)                                                \
  U(Uint32, dispatch_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_PAGED_ATTENTION_SCATTER_K_V_PROGRAM_CONFIG

using SparsePagedAttentionScatterKVProgram = ConfiguredProgram<SparsePagedAttentionScatterKVProgramShader>;

// Resolve, on device, the per-token values every sparse stage needs:
//   token_meta[t] = (batch_id, query_position, main_length, auxiliary_length,
//                    cumulative_query_start, past_length)
//
// Keeping this device-resident is what allows the op to run without any host
// readback of the metadata, selection, or block tables (a hard requirement for
// graph capture and for continuous-batching runtimes).
//
// Inputs (all read):
//   cumulative_sequence_length : (batch_size + 1,)  [S]
//   past_seqlens               : (batch_size,)      [S]
//   auxiliary_lengths          : (batch_size,)      [S]  (optional)
//
// Output (write):
//   token_meta : (token_count, 6)                   [S]
//
// Every int32 read is sanitized (negative values clamped, ranges ordered, main
// lengths clamped to max_num_blocks_per_seq * block_size) before it is written,
// so downstream shaders can convert to u32 safely and can never derive an
// unbounded candidate count from a caller-supplied past_seqlens value.
#define WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_CONFIG(F) F(bool, has_auxiliary_lengths_)

struct SparsePagedAttentionTokenMetaProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_auxiliary_lengths) {
    Config result{};
    result.has_auxiliary_lengths_ = bool(has_auxiliary_lengths);

    return result;
  }
  static constexpr std::string_view name = "SparsePagedAttentionTokenMeta";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                                     \
  U(Uint32, auxiliary_capacity)                                             \
  U(Uint32, max_main_positions)                                             \
  U(Uint32, dispatch_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_PAGED_ATTENTION_TOKEN_META_PROGRAM_CONFIG

using SparsePagedAttentionTokenMetaProgram = ConfiguredProgram<SparsePagedAttentionTokenMetaProgramShader>;

// Partial attention over the paged main cache.
//
// Depending on the op configuration this stage covers the local window, the
// selected main-cache positions, or their union (de-duplicated, because
// 'local_plus_selected' denotes a set union).
//
// Inputs (all read):
//   query            : (token_count, num_heads * head_size)              [T]
//   key_cache        : (num_blocks, block_size, kv_num_heads, head_size) [T]
//   value_cache      : same shape as key_cache                           [T]
//   token_meta       : (token_count, 6)                                  [S]
//   block_table      : (batch_size, max_num_blocks_per_seq)              [S]
//   selected_indices : (token_count, max_selected_entries)               [S] (optional)
//   selected_counts  : (token_count,)                                    [S] (optional)
//
// Output (write):
//   partial : (token_count, num_heads, head_size + 2)                    [float], or
//   output  : (token_count, num_heads * head_size)                       [T]
//
// The trailing two floats per (token, head) are the running softmax maximum
// and denominator, so the finalize stage can merge this partial state with the
// auxiliary partial state into one joint FP32 softmax.
//
// One workgroup handles one (token, head) pair and iterates the candidate list
// exactly `local_count + selected_count` times: bounded by the input shapes, by
// the number of positions the block table can address, and by the
// device-resident counts, and never truncated.
#define WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_CONFIG(F) \
  F(int, head_size_)                                         \
  F(bool, is_causal_)                                        \
  F(bool, use_local_window_)                                 \
  F(bool, use_selected_)                                     \
  F(bool, dedup_selected_)                                   \
  F(bool, use_slot_mapping_)                                 \
  F(bool, direct_output_)

struct SparsePagedAttentionMainProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_CONFIG);
  static Config MakeConfig(int head_size, bool is_causal, bool use_local_window, bool use_selected, bool dedup_selected,
                           bool use_slot_mapping, bool direct_output) {
    Config result{};
    result.head_size_ = int(head_size);
    result.is_causal_ = bool(is_causal);
    result.use_local_window_ = bool(use_local_window);
    result.use_selected_ = bool(use_selected);
    result.dedup_selected_ = bool(dedup_selected);
    result.use_slot_mapping_ = bool(use_slot_mapping);
    result.direct_output_ = bool(direct_output);

    return result;
  }
  static constexpr std::string_view name = "SparsePagedAttentionMain";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, num_heads)                                                \
  U(Uint32, kv_num_heads)                                             \
  U(Uint32, block_size)                                               \
  U(Uint32, num_blocks)                                               \
  U(Uint32, max_num_blocks_per_seq)                                   \
  U(Uint32, max_main_positions)                                       \
  U(Uint32, max_selected_entries)                                     \
  U(Uint32, local_window_size)                                        \
  U(Uint32, workgroup_count)                                          \
  U(Float32, scale)                                                   \
  U(Float32, softcap)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_PAGED_ATTENTION_MAIN_PROGRAM_CONFIG

using SparsePagedAttentionMainProgram = ConfiguredProgram<SparsePagedAttentionMainProgramShader>;

// Partial attention over the contiguous auxiliary cache.
//
// Inputs (all read):
//   query            : (token_count, num_heads * head_size)                [T]
//   auxiliary_key    : (batch_size, capacity, kv_heads_or_one, head_size)  [T]
//   auxiliary_value  : same shape as auxiliary_key               [T] (absent when K=V)
//   token_meta       : (token_count, 6)                                    [S]
//   selected_indices : (token_count, max_selected_entries)                 [S]
//   selected_counts  : (token_count,)                                      [S]
//
// Output (write):
//   partial : (token_count, num_heads, head_size + 2)                      [float], or
//   output  : (token_count, num_heads * head_size)                         [T]
//
// Selected positions are request-local rows in the auxiliary cache; entries
// that are negative or beyond the request's auxiliary_lengths value are
// ignored, matching the CUDA implementation.
#define WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_CONFIG(F) \
  F(int, head_size_)                                              \
  F(bool, auxiliary_kv_shared_)                                   \
  F(bool, direct_output_)

struct SparsePagedAttentionAuxiliaryProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_CONFIG);
  static Config MakeConfig(int head_size, bool auxiliary_kv_shared, bool direct_output) {
    Config result{};
    result.head_size_ = int(head_size);
    result.auxiliary_kv_shared_ = bool(auxiliary_kv_shared);
    result.direct_output_ = bool(direct_output);

    return result;
  }
  static constexpr std::string_view name = "SparsePagedAttentionAuxiliary";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, num_heads)                                                     \
  U(Uint32, kv_num_heads)                                                  \
  U(Uint32, auxiliary_capacity)                                            \
  U(Uint32, auxiliary_num_heads)                                           \
  U(Uint32, max_selected_entries)                                          \
  U(Uint32, workgroup_count)                                               \
  U(Float32, scale)                                                        \
  U(Float32, softcap)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_PAGED_ATTENTION_AUXILIARY_PROGRAM_CONFIG

using SparsePagedAttentionAuxiliaryProgram = ConfiguredProgram<SparsePagedAttentionAuxiliaryProgramShader>;

// Merge the partial softmax states into one joint FP32 softmax and write the
// activation-typed output.
//
// Inputs (read):
//   partial_main : (token_count, num_heads, head_size + 2) [float] (optional)
//   partial_aux  : (token_count, num_heads, head_size + 2) [float] (optional)
//   head_sink    : (num_heads,)                            [T]     (optional)
//
// Output (write):
//   output : (token_count, num_heads * head_size)          [T]
//
// The sink logit seeds the joint denominator exactly as the CUDA kernel does
// (running max = sink, running sum = 1), so the sink competes with both the
// main-cache and the auxiliary contributions in a single softmax.
#define WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_CONFIG(F) \
  F(bool, has_main_)                                             \
  F(bool, has_auxiliary_)                                        \
  F(bool, has_head_sink_)

struct SparsePagedAttentionFinalizeProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_main, bool has_auxiliary, bool has_head_sink) {
    Config result{};
    result.has_main_ = bool(has_main);
    result.has_auxiliary_ = bool(has_auxiliary);
    result.has_head_sink_ = bool(has_head_sink);

    return result;
  }
  static constexpr std::string_view name = "SparsePagedAttentionFinalize";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, num_heads)                                                    \
  U(Uint32, head_size)                                                    \
  U(Uint32, workgroup_count)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_PAGED_ATTENTION_FINALIZE_PROGRAM_CONFIG

using SparsePagedAttentionFinalizeProgram = ConfiguredProgram<SparsePagedAttentionFinalizeProgramShader>;

// com.microsoft.SparsePagedAttention for the WebGPU EP.
//
// Selection is produced by an external indexer and is consumed entirely on
// device. See docs/contrib_ops/webgpu/sparse_paged_attention.md for the
// support matrix and the staged execution plan.
class SparsePagedAttention final : public WebGpuKernel {
 public:
  explicit SparsePagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  bool is_causal_;
  bool do_rotary_;
  bool rotary_interleaved_;
  int rotary_offset_;
  float scale_;
  float softcap_;
  float qk_norm_epsilon_;
  bool has_explicit_scale_;
  std::string k_quant_type_;
  std::string v_quant_type_;
  SparseAttentionMode attention_mode_;
  SparseSelectedKvSource selected_kv_source_;
  bool auxiliary_kv_shared_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
