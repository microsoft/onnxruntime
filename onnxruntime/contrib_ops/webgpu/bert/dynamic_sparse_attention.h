// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_parameters.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_CONFIG(F) \
  F(bool, packed_qkv_)                                                  \
  F(bool, use_qk_norm_)                                                 \
  F(bool, do_rotary_)                                                   \
  F(bool, rotary_interleaved_)                                          \
  F(bool, has_position_ids_)

struct DynamicSparseAttentionPrepareQueryProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_CONFIG);
  static Config MakeConfig(bool packed_qkv, bool use_qk_norm, bool do_rotary, bool rotary_interleaved, bool has_position_ids) {
    Config result{};
    result.packed_qkv_ = bool{packed_qkv};
    result.use_qk_norm_ = bool{use_qk_norm};
    result.do_rotary_ = bool{do_rotary};
    result.rotary_interleaved_ = bool{rotary_interleaved};
    result.has_position_ids_ = bool{has_position_ids};

    return result;
  }
  static constexpr std::string_view name = "DynamicSparseAttentionPrepareQuery";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, sequence_length)                                                     \
  U(Uint32, num_heads)                                                           \
  U(Uint32, head_size)                                                           \
  U(Uint32, query_hidden_size)                                                   \
  U(Uint32, packed_stride)                                                       \
  U(Uint32, rotary_dim)                                                          \
  U(Uint32, rotary_offset)                                                       \
  U(Uint32, rotary_max_position)                                                 \
  U(Float32, qk_norm_epsilon)                                                    \
  U(Uint32, num_workgroups)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_CONFIG

using DynamicSparseAttentionPrepareQueryProgram = ConfiguredProgram<DynamicSparseAttentionPrepareQueryProgramShader>;

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_CONFIG(F) \
  F(bool, initialize_key_)                                                 \
  F(bool, initialize_value_)                                               \
  F(bool, has_past_key_)                                                   \
  F(bool, has_past_value_)

struct DynamicSparseAttentionInitializeCacheProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_CONFIG);
  static Config MakeConfig(bool initialize_key, bool initialize_value, bool has_past_key, bool has_past_value) {
    Config result{};
    result.initialize_key_ = bool{initialize_key};
    result.initialize_value_ = bool{initialize_value};
    result.has_past_key_ = bool{has_past_key};
    result.has_past_value_ = bool{has_past_value};

    return result;
  }
  static constexpr std::string_view name = "DynamicSparseAttentionInitializeCache";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, dispatch_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_CONFIG

using DynamicSparseAttentionInitializeCacheProgram =
    ConfiguredProgram<DynamicSparseAttentionInitializeCacheProgramShader>;

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_CONFIG(F) \
  F(bool, packed_qkv_)                                              \
  F(bool, use_qk_norm_)                                             \
  F(bool, do_rotary_)                                               \
  F(bool, rotary_interleaved_)                                      \
  F(bool, has_position_ids_)

struct DynamicSparseAttentionAppendKvProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_CONFIG);
  static Config MakeConfig(bool packed_qkv, bool use_qk_norm, bool do_rotary, bool rotary_interleaved, bool has_position_ids) {
    Config result{};
    result.packed_qkv_ = bool{packed_qkv};
    result.use_qk_norm_ = bool{use_qk_norm};
    result.do_rotary_ = bool{do_rotary};
    result.rotary_interleaved_ = bool{rotary_interleaved};
    result.has_position_ids_ = bool{has_position_ids};

    return result;
  }
  static constexpr std::string_view name = "DynamicSparseAttentionAppendKv";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, sequence_length)                                                 \
  U(Uint32, kv_num_heads)                                                    \
  U(Uint32, head_size)                                                       \
  U(Uint32, query_hidden_size)                                               \
  U(Uint32, kv_hidden_size)                                                  \
  U(Uint32, cache_capacity)                                                  \
  U(Uint32, packed_stride)                                                   \
  U(Uint32, rotary_dim)                                                      \
  U(Uint32, rotary_offset)                                                   \
  U(Uint32, rotary_max_position)                                             \
  U(Float32, qk_norm_epsilon)                                                \
  U(Uint32, num_workgroups)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_CONFIG

using DynamicSparseAttentionAppendKvProgram = ConfiguredProgram<DynamicSparseAttentionAppendKvProgramShader>;

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_CONFIG(F) \
  F(bool, has_selection_)                                 \
  F(bool, local_plus_selected_)                           \
  F(bool, selected_from_auxiliary_)                       \
  F(bool, has_auxiliary_value_)                           \
  F(bool, has_head_sink_)                                 \
  F(bool, use_smooth_softmax_)

struct DynamicSparseAttentionProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_selection, bool local_plus_selected, bool selected_from_auxiliary, bool has_auxiliary_value,
                           bool has_head_sink, bool use_smooth_softmax) {
    Config result{};
    result.has_selection_ = bool{has_selection};
    result.local_plus_selected_ = bool{local_plus_selected};
    result.selected_from_auxiliary_ = bool{selected_from_auxiliary};
    result.has_auxiliary_value_ = bool{has_auxiliary_value};
    result.has_head_sink_ = bool{has_head_sink};
    result.use_smooth_softmax_ = bool{use_smooth_softmax};

    return result;
  }
  static constexpr std::string_view name = "DynamicSparseAttention";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, sequence_length)                                       \
  U(Uint32, num_heads)                                             \
  U(Uint32, kv_num_heads)                                          \
  U(Uint32, head_size)                                             \
  U(Uint32, cache_capacity)                                        \
  U(Uint32, auxiliary_sequence_length)                             \
  U(Uint32, max_selected)                                          \
  U(Uint32, local_window_size)                                     \
  U(Float32, scale)                                                \
  U(Uint32, num_workgroups)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_CONFIG

using DynamicSparseAttentionProgram = ConfiguredProgram<DynamicSparseAttentionProgramShader>;

class DynamicSparseAttention final : public WebGpuKernel {
 public:
  explicit DynamicSparseAttention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  int rotary_offset_;
  float scale_;
  bool has_scale_;
  float qk_norm_epsilon_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool use_smooth_softmax_;
  bool auxiliary_kv_shared_;
  DynamicSparseAttentionMode attention_mode_;
  DynamicSparseAttentionKvSource selected_kv_source_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
