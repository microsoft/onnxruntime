// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class DynamicSparseAttentionPrepareQueryProgram final : public Program<DynamicSparseAttentionPrepareQueryProgram> {
 public:
  DynamicSparseAttentionPrepareQueryProgram(bool packed_qkv, bool use_qk_norm, bool do_rotary,
                                            bool rotary_interleaved, bool has_position_ids)
      : Program{"DynamicSparseAttentionPrepareQuery"},
        packed_qkv_{packed_qkv},
        use_qk_norm_{use_qk_norm},
        do_rotary_{do_rotary},
        rotary_interleaved_{rotary_interleaved},
        has_position_ids_{has_position_ids} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

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
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool packed_qkv_;
  bool use_qk_norm_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool has_position_ids_;
};

class DynamicSparseAttentionInitializeCacheProgram final
    : public Program<DynamicSparseAttentionInitializeCacheProgram> {
 public:
  DynamicSparseAttentionInitializeCacheProgram(bool initialize_key, bool initialize_value,
                                               bool has_past_key, bool has_past_value)
      : Program{"DynamicSparseAttentionInitializeCache"},
        initialize_key_{initialize_key},
        initialize_value_{initialize_value},
        has_past_key_{has_past_key},
        has_past_value_{has_past_value} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool initialize_key_;
  bool initialize_value_;
  bool has_past_key_;
  bool has_past_value_;
};

class DynamicSparseAttentionAppendKvProgram final : public Program<DynamicSparseAttentionAppendKvProgram> {
 public:
  DynamicSparseAttentionAppendKvProgram(bool packed_qkv, bool use_qk_norm, bool do_rotary,
                                        bool rotary_interleaved, bool has_position_ids)
      : Program{"DynamicSparseAttentionAppendKv"},
        packed_qkv_{packed_qkv},
        use_qk_norm_{use_qk_norm},
        do_rotary_{do_rotary},
        rotary_interleaved_{rotary_interleaved},
        has_position_ids_{has_position_ids} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

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
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool packed_qkv_;
  bool use_qk_norm_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool has_position_ids_;
};

class DynamicSparseAttentionProgram final : public Program<DynamicSparseAttentionProgram> {
 public:
  DynamicSparseAttentionProgram(bool has_selection, bool local_plus_selected, bool selected_from_auxiliary,
                                bool has_auxiliary_value, bool has_head_sink, bool use_smooth_softmax)
      : Program{"DynamicSparseAttention"},
        has_selection_{has_selection},
        local_plus_selected_{local_plus_selected},
        selected_from_auxiliary_{selected_from_auxiliary},
        has_auxiliary_value_{has_auxiliary_value},
        has_head_sink_{has_head_sink},
        use_smooth_softmax_{use_smooth_softmax} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"cache_capacity", ProgramUniformVariableDataType::Uint32},
      {"auxiliary_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"max_selected", ProgramUniformVariableDataType::Uint32},
      {"local_window_size", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  bool has_selection_;
  bool local_plus_selected_;
  bool selected_from_auxiliary_;
  bool has_auxiliary_value_;
  bool has_head_sink_;
  bool use_smooth_softmax_;
};

class DynamicSparseAttention final : public WebGpuKernel {
 public:
  explicit DynamicSparseAttention(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  int rotary_offset_;
  float scale_;
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
