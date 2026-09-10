// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

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
class SparsePagedAttentionScatterKVProgram final
    : public Program<SparsePagedAttentionScatterKVProgram> {
 public:
  SparsePagedAttentionScatterKVProgram() : Program{"SparsePagedAttentionScatterKV"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"cache_slot_count", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
};

// Resolve, on device, the per-token values every sparse stage needs:
//   token_meta[t] = (batch_id, query_position, main_length, auxiliary_length)
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
//   token_meta : (token_count, 4)                   [S]
//
// Every int32 read is sanitized (negative values clamped, ranges ordered)
// before it is written, so downstream shaders can convert to u32 safely.
class SparsePagedAttentionTokenMetaProgram final
    : public Program<SparsePagedAttentionTokenMetaProgram> {
 public:
  explicit SparsePagedAttentionTokenMetaProgram(bool has_auxiliary_lengths)
      : Program{"SparsePagedAttentionTokenMeta"},
        has_auxiliary_lengths_(has_auxiliary_lengths) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"auxiliary_capacity", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_auxiliary_lengths_;
};

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
//   token_meta       : (token_count, 4)                                  [S]
//   block_table      : (batch_size, max_num_blocks_per_seq)              [S]
//   selected_indices : (token_count, max_selected_entries)               [S] (optional)
//   selected_counts  : (token_count,)                                    [S] (optional)
//
// Output (write):
//   partial : (token_count, num_heads, head_size + 2)                    [float]
//
// The trailing two floats per (token, head) are the running softmax maximum
// and denominator, so the finalize stage can merge this partial state with the
// auxiliary partial state into one joint FP32 softmax.
//
// One workgroup handles one (token, head) pair and iterates the candidate list
// exactly `local_count + selected_count` times: bounded by the input shapes and
// device-resident counts, and never truncated.
class SparsePagedAttentionMainProgram final
    : public Program<SparsePagedAttentionMainProgram> {
 public:
  SparsePagedAttentionMainProgram(int head_size, bool is_causal, bool use_local_window,
                                  bool use_selected, bool dedup_selected)
      : Program{"SparsePagedAttentionMain"},
        head_size_(head_size),
        is_causal_(is_causal),
        use_local_window_(use_local_window),
        use_selected_(use_selected),
        dedup_selected_(dedup_selected) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"num_blocks", ProgramUniformVariableDataType::Uint32},
      {"max_num_blocks_per_seq", ProgramUniformVariableDataType::Uint32},
      {"max_selected_entries", ProgramUniformVariableDataType::Uint32},
      {"local_window_size", ProgramUniformVariableDataType::Uint32},
      {"workgroup_count", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"softcap", ProgramUniformVariableDataType::Float32});

 private:
  int head_size_;
  bool is_causal_;
  bool use_local_window_;
  bool use_selected_;
  bool dedup_selected_;
};

// Partial attention over the contiguous auxiliary cache.
//
// Inputs (all read):
//   query            : (token_count, num_heads * head_size)                [T]
//   auxiliary_key    : (batch_size, capacity, kv_heads_or_one, head_size)  [T_AUX]
//   auxiliary_value  : same shape as auxiliary_key           [T_AUX] (absent when K=V)
//   token_meta       : (token_count, 4)                                    [S]
//   selected_indices : (token_count, max_selected_entries)                 [S]
//   selected_counts  : (token_count,)                                      [S]
//
// Output (write):
//   partial : (token_count, num_heads, head_size + 2)                      [float]
//
// Selected positions are request-local rows in the auxiliary cache; entries
// that are negative or beyond the request's auxiliary_lengths value are
// ignored, matching the CUDA implementation.
class SparsePagedAttentionAuxiliaryProgram final
    : public Program<SparsePagedAttentionAuxiliaryProgram> {
 public:
  SparsePagedAttentionAuxiliaryProgram(int head_size, bool auxiliary_kv_shared)
      : Program{"SparsePagedAttentionAuxiliary"},
        head_size_(head_size),
        auxiliary_kv_shared_(auxiliary_kv_shared) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"auxiliary_capacity", ProgramUniformVariableDataType::Uint32},
      {"auxiliary_num_heads", ProgramUniformVariableDataType::Uint32},
      {"max_selected_entries", ProgramUniformVariableDataType::Uint32},
      {"workgroup_count", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"softcap", ProgramUniformVariableDataType::Float32});

 private:
  int head_size_;
  bool auxiliary_kv_shared_;
};

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
class SparsePagedAttentionFinalizeProgram final
    : public Program<SparsePagedAttentionFinalizeProgram> {
 public:
  SparsePagedAttentionFinalizeProgram(bool has_main, bool has_auxiliary, bool has_head_sink)
      : Program{"SparsePagedAttentionFinalize"},
        has_main_(has_main),
        has_auxiliary_(has_auxiliary),
        has_head_sink_(has_head_sink) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_main_;
  bool has_auxiliary_;
  bool has_head_sink_;
};

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
