// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias) {
    Config result{};
    result.has_bias_ = bool(has_bias);

    return result;
  }
  static constexpr std::string_view name = "TransferBSDToBNSH";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, data_size)                                              \
  U(Uint32, batch_offset)                                           \
  U(Uint32, sequence_offset)                                        \
  U(Uint32, head_offset)                                            \
  U(Uint32, bias_offset)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TRANSFER_B_S_D_TO_B_N_S_H_PROGRAM_CONFIG

using TransferBSDToBNSHProgram = ConfiguredProgram<TransferBSDToBNSHProgramShader>;

#define WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG(F)

struct SplitPackedQKVProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "SplitPackedQKV";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, input_size)                                      \
  U(Uint32, hidden_size)                                     \
  U(Uint32, kv_hidden_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPLIT_PACKED_Q_K_V_PROGRAM_CONFIG

using SplitPackedQKVProgram = ConfiguredProgram<SplitPackedQKVProgramShader>;

#define WEBGPU_ATTENTION_PROBS_PROGRAM_CONFIG(F) \
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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_ATTENTION_PROBS_PROGRAM_CONFIG);
  static onnxruntime::webgpu::ShaderProgramSetup<Config> MakeConfig(const std::string& kernel_name, bool feed_past_key, bool has_present_key, bool has_attention_bias,
                                                                    int tile_size, int components, bool is_first_prompt, bool has_seqlen_k = false,
                                                                    bool past_present_share_buffer = false, bool is_unidirectional = false) {
    Config result{};

    result.feed_past_key_ = bool(feed_past_key);
    result.has_present_key_ = bool(has_present_key);
    result.has_attention_bias_ = bool(has_attention_bias);
    result.tile_size_ = int(tile_size);
    result.components_ = int(components);
    result.has_seqlen_k_ = bool(has_seqlen_k);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.is_first_prompt_ = bool(is_first_prompt);
    result.is_unidirectional_ = bool(is_unidirectional);

    return {std::move(result), kernel_name};
  }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_ATTENTION_PROBS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, M)                                            \
  U(Uint32, K)                                            \
  U(Uint32, N)                                            \
  U(Uint32, num_heads)                                    \
  U(Uint32, head_size)                                    \
  U(Float32, alpha)                                       \
  U(Uint32, past_sequence_length)                         \
  U(Uint32, kv_sequence_length)                           \
  U(Uint32, present_sequence_length)                      \
  U(Uint32, n_reps)                                       \
  U(Uint32, is_first_prompt)                              \
  U(Uint32, num_total_seq_length_tile)                    \
  U(Uint32, num_seq_length_tile)                          \
  U(Uint32, attn_bias_dim0)                               \
  U(Uint32, attn_bias_dim1)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_ATTENTION_PROBS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_ATTENTION_PROBS_PROGRAM_SHADER_UNIFORMS

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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_CONFIG);
  static Config MakeConfig(int work_group_size, int components, bool use_smooth_softmax, bool has_seqlen_k, bool has_head_sink,
                           int local_window_size) {
    Config result{};
    result.work_group_size_ = int(work_group_size);
    result.components_ = int(components);
    result.use_smooth_softmax_ = bool(use_smooth_softmax);
    result.has_seqlen_k_ = bool(has_seqlen_k);
    result.has_head_sink_ = bool(has_head_sink);
    result.local_window_size_ = int(local_window_size);

    return result;
  }
  static constexpr std::string_view name = "InPlaceSoftmax";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                    \
  U(Uint32, num_heads)                                     \
  U(Uint32, past_sequence_length)                          \
  U(Uint32, kv_sequence_length)                            \
  U(Uint32, present_sequence_length)                       \
  U(Uint32, sequence_length)                               \
  U(Uint32, total_sequence_length_comp)                    \
  U(Uint32, elements_per_thread)                           \
  U(Uint32, is_first_prompt)                               \
  U(Uint32, local_window_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_IN_PLACE_SOFTMAX_PROGRAM_CONFIG

using InPlaceSoftmaxProgram = ConfiguredProgram<InPlaceSoftmaxProgramShader>;

#define WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG(F) \
  F(bool, feed_past_value_)                         \
  F(bool, has_present_value_)                       \
  F(int, tile_size_)                                \
  F(bool, seqlen_k_)                                \
  F(bool, past_present_share_buffer_)               \
  F(bool, is_first_prompt_)

struct VxAttentionScoreProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG);
  static onnxruntime::webgpu::ShaderProgramSetup<Config> MakeConfig(const std::string& kernel_name, bool feed_past_value, bool has_present_value, int tile_size,
                                                                    bool is_first_prompt, const Tensor* seqlen_k = nullptr, bool past_present_share_buffer = false) {
    Config result{};

    result.feed_past_value_ = bool(feed_past_value);
    result.has_present_value_ = bool(has_present_value);
    result.tile_size_ = int(tile_size);
    result.seqlen_k_ = bool(seqlen_k != nullptr);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.is_first_prompt_ = bool(is_first_prompt);

    return {std::move(result), kernel_name};
  }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_VX_ATTENTION_SCORE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, M)                                               \
  U(Uint32, K)                                               \
  U(Uint32, N)                                               \
  U(Uint32, num_heads)                                       \
  U(Uint32, head_size)                                       \
  U(Uint32, v_hidden_size)                                   \
  U(Uint32, past_sequence_length)                            \
  U(Uint32, kv_sequence_length)                              \
  U(Uint32, present_sequence_length)                         \
  U(Uint32, n_reps)                                          \
  U(Uint32, is_first_prompt)                                 \
  U(Uint32, num_head_size_tile)                              \
  U(Uint32, num_seq_length_tile)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VX_ATTENTION_SCORE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VX_ATTENTION_SCORE_PROGRAM_SHADER_UNIFORMS

  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"TILE_SIZE", ProgramConstantDataType::Uint32});
};
#undef WEBGPU_VX_ATTENTION_SCORE_PROGRAM_CONFIG

using VxAttentionScoreProgram = ConfiguredProgram<VxAttentionScoreProgramShader>;

class Attention final : public WebGpuKernel, public onnxruntime::contrib::AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  mutable MatMulComputeDispatcher matmul_compute_dispatcher_;
  bool weights_are_constant_ = false;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
