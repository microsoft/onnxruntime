// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/webgpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_CONFIG(F) F(bool, has_bias_)

struct TransferBSDToBNSHProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_CONFIG);
    Config(bool has_bias) : has_bias_(has_bias) {}
  };
  static constexpr std::string_view name = "TransferBSDToBNSH";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32},
                                          {"batch_offset", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_offset", ProgramUniformVariableDataType::Uint32},
                                          {"head_offset", ProgramUniformVariableDataType::Uint32},
                                          {"bias_offset", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_CONFIG

using TransferBSDToBNSHProgram = ConfiguredProgram<TransferBSDToBNSHProgramShader>;

#define WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG(F)

struct SplitPackedQKVProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "SplitPackedQKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_size", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kv_hidden_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG

using SplitPackedQKVProgram = ConfiguredProgram<SplitPackedQKVProgramShader>;

#define WEBGPU_ATTENTION_PROBS_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                  \
  F(bool, feed_past_key_)                        \
  F(bool, has_present_key_)                      \
  F(bool, has_attention_bias_)                   \
  F(int, tile_size_)                             \
  F(int, components_)                            \
  F(bool, has_seqlen_k_)                         \
  F(bool, past_present_share_buffer_)            \
  F(bool, is_first_prompt_)                      \
  F(bool, is_unidirectional_)

struct AttentionProbsProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_ATTENTION_PROBS_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool feed_past_key, bool has_present_key, bool has_attention_bias,
           int tile_size, int components, bool is_first_prompt, bool has_seqlen_k = false,
           bool past_present_share_buffer = false, bool is_unidirectional = false)
        : program_name_{kernel_name},
          feed_past_key_(feed_past_key),
          has_present_key_(has_present_key),
          has_attention_bias_(has_attention_bias),
          tile_size_(tile_size),
          components_(components),
          has_seqlen_k_(has_seqlen_k),
          past_present_share_buffer_(past_present_share_buffer),
          is_first_prompt_(is_first_prompt),
          is_unidirectional_(is_unidirectional) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"head_size", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"is_first_prompt", ProgramUniformVariableDataType::Uint32},
                                          {"num_total_seq_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_seq_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32});

  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"TILE_SIZE", ProgramConstantDataType::Uint32});
};
#undef WEBGPU_ATTENTION_PROBS_PROGRAM_CONFIG

using AttentionProbsProgram = ConfiguredProgram<AttentionProbsProgramShader>;

#define WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_CONFIG(F) \
  F(int, work_group_size_)                        \
  F(int, components_)                             \
  F(bool, use_smooth_softmax_)                    \
  F(bool, has_seqlen_k_)                          \
  F(bool, has_head_sink_)                         \
  F(int, local_window_size_)

struct InPlaceSoftmaxProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_CONFIG);
    Config(int work_group_size, int components, bool use_smooth_softmax, bool has_seqlen_k, bool has_head_sink,
           int local_window_size)
        : work_group_size_(work_group_size),
          components_(components),
          use_smooth_softmax_(use_smooth_softmax),
          has_seqlen_k_(has_seqlen_k),
          has_head_sink_(has_head_sink),
          local_window_size_(local_window_size) {}
  };
  static constexpr std::string_view name = "InPlaceSoftmax";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length_comp", ProgramUniformVariableDataType::Uint32},
                                          {"elements_per_thread", ProgramUniformVariableDataType::Uint32},
                                          {"is_first_prompt", ProgramUniformVariableDataType::Uint32},
                                          {"local_window_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_CONFIG

using InPlaceSoftmaxProgram = ConfiguredProgram<InPlaceSoftmaxProgramShader>;

#define WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                     \
  F(bool, feed_past_value_)                         \
  F(bool, has_present_value_)                       \
  F(int, tile_size_)                                \
  F(bool, seqlen_k_)                                \
  F(bool, past_present_share_buffer_)               \
  F(bool, is_first_prompt_)

struct VxAttentionScoreProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, bool feed_past_value, bool has_present_value, int tile_size,
           bool is_first_prompt, const Tensor* seqlen_k = nullptr, bool past_present_share_buffer = false)
        : program_name_{kernel_name},
          feed_past_value_(feed_past_value),
          has_present_value_(has_present_value),
          tile_size_(tile_size),
          seqlen_k_(seqlen_k != nullptr),
          past_present_share_buffer_(past_present_share_buffer),
          is_first_prompt_(is_first_prompt) {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"head_size", ProgramUniformVariableDataType::Uint32},
                                          {"v_hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"is_first_prompt", ProgramUniformVariableDataType::Uint32},
                                          {"num_head_size_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_seq_length_tile", ProgramUniformVariableDataType::Uint32});

  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"TILE_SIZE", ProgramConstantDataType::Uint32});
};
#undef WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG

using VxAttentionScoreProgram = ConfiguredProgram<VxAttentionScoreProgramShader>;

class Attention final : public WebGpuKernel, public onnxruntime::contrib::AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  mutable MatMulOptImplCache matmul_compute_cache_;
  bool weights_are_constant_ = false;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
