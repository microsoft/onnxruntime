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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_CONFIG);
    Config(bool packed_qkv, bool use_qk_norm, bool do_rotary, bool rotary_interleaved, bool has_position_ids)
        : packed_qkv_{packed_qkv},
          use_qk_norm_{use_qk_norm},
          do_rotary_{do_rotary},
          rotary_interleaved_{rotary_interleaved},
          has_position_ids_{has_position_ids} {}
  };
  static constexpr std::string_view name = "DynamicSparseAttentionPrepareQuery";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"packed_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"rotary_offset", ProgramUniformVariableDataType::Uint32},
      {"rotary_max_position", ProgramUniformVariableDataType::Uint32},
      {"qk_norm_epsilon", ProgramUniformVariableDataType::Float32},
      {"num_workgroups", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_DYNAMIC_SPARSE_ATTENTION_PREPARE_QUERY_PROGRAM_CONFIG

using DynamicSparseAttentionPrepareQueryProgram = ConfiguredProgram<DynamicSparseAttentionPrepareQueryProgramShader>;

#define WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_CONFIG(F) \
  F(bool, initialize_key_)                                                 \
  F(bool, initialize_value_)                                               \
  F(bool, has_past_key_)                                                   \
  F(bool, has_past_value_)

struct DynamicSparseAttentionInitializeCacheProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_INITIALIZE_CACHE_PROGRAM_CONFIG);
    Config(bool initialize_key, bool initialize_value, bool has_past_key, bool has_past_value)
        : initialize_key_{initialize_key},
          initialize_value_{initialize_value},
          has_past_key_{has_past_key},
          has_past_value_{has_past_value} {}
  };
  static constexpr std::string_view name = "DynamicSparseAttentionInitializeCache";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});
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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_APPEND_KV_PROGRAM_CONFIG);
    Config(bool packed_qkv, bool use_qk_norm, bool do_rotary, bool rotary_interleaved, bool has_position_ids)
        : packed_qkv_{packed_qkv},
          use_qk_norm_{use_qk_norm},
          do_rotary_{do_rotary},
          rotary_interleaved_{rotary_interleaved},
          has_position_ids_{has_position_ids} {}
  };
  static constexpr std::string_view name = "DynamicSparseAttentionAppendKv";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"cache_capacity", ProgramUniformVariableDataType::Uint32},
      {"packed_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"rotary_offset", ProgramUniformVariableDataType::Uint32},
      {"rotary_max_position", ProgramUniformVariableDataType::Uint32},
      {"qk_norm_epsilon", ProgramUniformVariableDataType::Float32},
      {"num_workgroups", ProgramUniformVariableDataType::Uint32});
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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DYNAMIC_SPARSE_ATTENTION_PROGRAM_CONFIG);
    Config(bool has_selection, bool local_plus_selected, bool selected_from_auxiliary, bool has_auxiliary_value,
           bool has_head_sink, bool use_smooth_softmax)
        : has_selection_{has_selection},
          local_plus_selected_{local_plus_selected},
          selected_from_auxiliary_{selected_from_auxiliary},
          has_auxiliary_value_{has_auxiliary_value},
          has_head_sink_{has_head_sink},
          use_smooth_softmax_{use_smooth_softmax} {}
  };
  static constexpr std::string_view name = "DynamicSparseAttention";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"cache_capacity", ProgramUniformVariableDataType::Uint32},
      {"auxiliary_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"max_selected", ProgramUniformVariableDataType::Uint32},
      {"local_window_size", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"num_workgroups", ProgramUniformVariableDataType::Uint32});
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
