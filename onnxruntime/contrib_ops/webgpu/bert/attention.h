// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class TransferBSDToBNSHProgram final : public Program<TransferBSDToBNSHProgram> {
 public:
  TransferBSDToBNSHProgram(bool has_bias) : Program{"TransferBSDToBNSH"}, has_bias_(has_bias) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32},
                                          {"batch_offset", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_offset", ProgramUniformVariableDataType::Uint32},
                                          {"head_offset", ProgramUniformVariableDataType::Uint32},
                                          {"bias_offset", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
};

class AttentionProbsProgram final : public Program<AttentionProbsProgram> {
 public:
  AttentionProbsProgram(const std::string& kernel_name, bool feed_past_key, bool has_present_key,
                        bool has_attention_bias, int tile_size, int components, bool is_first_prompt, int n_reps = 1, const Tensor* seqlen_k = nullptr, bool past_present_share_buffer = false)
      : Program{kernel_name}, feed_past_key_(feed_past_key), has_present_key_(has_present_key), has_attention_bias_(has_attention_bias), tile_size_(tile_size), components_(components), n_reps_(n_reps), seqlen_k_(seqlen_k), past_present_share_buffer_(past_present_share_buffer), is_first_prompt_(is_first_prompt) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"head_size", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32});

  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"TILE_SIZE", ProgramConstantDataType::Uint32});

 private:
  bool feed_past_key_;
  bool has_present_key_;
  bool has_attention_bias_;
  int tile_size_;
  int components_;
  int n_reps_;
  const Tensor* seqlen_k_;
  bool past_present_share_buffer_;
  bool is_first_prompt_;
};

class InPlaceSoftmaxProgram final : public Program<InPlaceSoftmaxProgram> {
 public:
  InPlaceSoftmaxProgram(const std::string& kernel_name, int work_group_size, int components, bool is_first_prompt, const Tensor* seqlen_k = nullptr)
      : Program{kernel_name}, work_group_size_(work_group_size), components_(components), seqlen_k_(seqlen_k), is_first_prompt_(is_first_prompt) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length_comp", ProgramUniformVariableDataType::Uint32},
                                          {"elements_per_thread", ProgramUniformVariableDataType::Uint32});

 private:
  int work_group_size_;
  int components_;
  const Tensor* seqlen_k_;
  bool is_first_prompt_;
};

class VxAttentionScoreProgram final : public Program<VxAttentionScoreProgram> {
 public:
  VxAttentionScoreProgram(const std::string& kernel_name, bool feed_past_value, bool has_present_value, int tile_size, bool is_first_prompt, int n_reps = 1, const Tensor* seqlen_k = nullptr, bool past_present_share_buffer = false)
      : Program{kernel_name}, feed_past_value_(feed_past_value), has_present_value_(has_present_value), tile_size_(tile_size), n_reps_(n_reps), seqlen_k_(seqlen_k), past_present_share_buffer_(past_present_share_buffer), is_first_prompt_(is_first_prompt) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"head_size", ProgramUniformVariableDataType::Uint32},
                                          {"v_hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32});

  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"TILE_SIZE", ProgramConstantDataType::Uint32});

 private:
  bool feed_past_value_;
  bool has_present_value_;
  int tile_size_;
  int n_reps_;
  const Tensor* seqlen_k_;
  bool past_present_share_buffer_;
  bool is_first_prompt_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
