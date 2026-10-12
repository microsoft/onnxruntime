// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

enum class GatedDeltaNetUpdateRule {
  Linear,
  Gated,
  Delta,
  GatedDelta,
  Invalid,
};

struct GatedDeltaNetParallelPrefillPlan {
  uint32_t chunks_per_pass;
  uint64_t workspace_bytes;
};

inline uint64_t GatedDeltaNetChunkCapacity(uint64_t total_tokens, uint64_t batch,
                                           uint32_t chunk_size, bool ragged) {
  if (ragged) return total_tokens / chunk_size + batch;
  return batch * ((total_tokens / batch + chunk_size - 1) / chunk_size);
}

// The prepare and output passes each keep one state-shaped tile per live chunk. Two
// additional state tiles ping-pong the recurrent carry between passes.
inline std::optional<GatedDeltaNetParallelPrefillPlan> SelectGatedDeltaNetParallelPrefillPlan(
    uint64_t state_elements, uint32_t total_chunks, uint64_t workspace_cap_bytes = 64ull << 20) {
  if (total_chunks < 2 || state_elements == 0 ||
      state_elements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return std::nullopt;
  }

  const uint64_t state_bytes = state_elements * sizeof(float);
  if (state_bytes > workspace_cap_bytes / 2) {
    return std::nullopt;
  }

  const uint64_t fixed_bytes = 2 * state_bytes;
  const uint64_t bytes_per_chunk = 2 * state_bytes;
  if (bytes_per_chunk == 0 || fixed_bytes >= workspace_cap_bytes) {
    return std::nullopt;
  }

  const uint64_t max_chunks = (workspace_cap_bytes - fixed_bytes) / bytes_per_chunk;
  if (max_chunks == 0) {
    return std::nullopt;
  }

  const uint32_t chunks_per_pass =
      static_cast<uint32_t>(std::min<uint64_t>(total_chunks, max_chunks));
  const uint64_t workspace_bytes = fixed_bytes + chunks_per_pass * bytes_per_chunk;
  return GatedDeltaNetParallelPrefillPlan{chunks_per_pass, workspace_bytes};
}

#define WEBGPU_GATED_DELTA_NET_PROGRAM_CONFIG(F) \
  F(GatedDeltaNetUpdateRule, update_rule_)       \
  F(bool, has_cu_seqlens_)                       \
  F(bool, has_initial_state_)                    \
  F(bool, initial_state_in_final_state_)         \
  F(bool, output_final_state_)                   \
  F(bool, qwen_gate_)                            \
  F(bool, sigmoid_beta_)                         \
  F(bool, qk_l2_norm_)                           \
  F(bool, use_packed_params_)                    \
  F(bool, capture_state_updates_)                \
  F(bool, vectorized_value_io_)

struct GatedDeltaNetProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_PROGRAM_CONFIG);
  static Config MakeConfig(GatedDeltaNetUpdateRule update_rule, bool has_cu_seqlens, bool has_initial_state,
                           bool initial_state_in_final_state, bool output_final_state, bool qwen_gate, bool sigmoid_beta,
                           bool qk_l2_norm, bool use_packed_params, bool capture_state_updates, bool vectorized_value_io) {
    Config result{};
    result.update_rule_ = GatedDeltaNetUpdateRule(update_rule);
    result.has_cu_seqlens_ = bool(has_cu_seqlens);
    result.has_initial_state_ = bool(has_initial_state);
    result.initial_state_in_final_state_ = bool(initial_state_in_final_state);
    result.output_final_state_ = bool(output_final_state);
    result.qwen_gate_ = bool(qwen_gate);
    result.sigmoid_beta_ = bool(sigmoid_beta);
    result.qk_l2_norm_ = bool(qk_l2_norm);
    result.use_packed_params_ = bool(use_packed_params);
    result.capture_state_updates_ = bool(capture_state_updates);
    result.vectorized_value_io_ = bool(vectorized_value_io);

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNet";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                 \
  U(Uint32, batch_size)                                   \
  U(Uint32, num_heads_q)                                  \
  U(Uint32, num_heads_v)                                  \
  U(Uint32, head_size_qk)                                 \
  U(Uint32, head_size_v)                                  \
  U(Uint32, state_update_capacity)                        \
  U(Float32, scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_PROGRAM_CONFIG

using GatedDeltaNetProgram = ConfiguredProgram<GatedDeltaNetProgramShader>;

#define WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG(F)

struct GatedDeltaNetClearProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetClear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, element_count)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG

using GatedDeltaNetClearProgram = ConfiguredProgram<GatedDeltaNetClearProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG(F)

struct GatedDeltaNetPrefillPrepareProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetPrefillPrepare";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                                 \
  U(Uint32, batch_size)                                                   \
  U(Uint32, num_heads_q)                                                  \
  U(Uint32, num_heads_v)                                                  \
  U(Uint32, head_size_qk)                                                 \
  U(Uint32, head_size_v)                                                  \
  U(Uint32, chunk_size)                                                   \
  U(Uint32, chunk_base)                                                   \
  U(Uint32, chunks_in_pass)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG

using GatedDeltaNetPrefillPrepareProgram = ConfiguredProgram<GatedDeltaNetPrefillPrepareProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG(F) \
  F(bool, has_initial_state_)                                 \
  F(bool, has_carry_state_)                                   \
  F(bool, output_final_state_)

struct GatedDeltaNetPrefillScanProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_initial_state, bool has_carry_state, bool output_final_state) {
    Config result{};
    result.has_initial_state_ = bool(has_initial_state);
    result.has_carry_state_ = bool(has_carry_state);
    result.output_final_state_ = bool(output_final_state);

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetPrefillScan";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                                \
  U(Uint32, num_heads_v)                                               \
  U(Uint32, head_size_qk)                                              \
  U(Uint32, head_size_v)                                               \
  U(Uint32, chunks_in_pass)                                            \
  U(Uint32, is_last_pass)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG

using GatedDeltaNetPrefillScanProgram = ConfiguredProgram<GatedDeltaNetPrefillScanProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG(F)

struct GatedDeltaNetPrefillOutputProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetPrefillOutput";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                                \
  U(Uint32, batch_size)                                                  \
  U(Uint32, num_heads_q)                                                 \
  U(Uint32, num_heads_v)                                                 \
  U(Uint32, head_size_qk)                                                \
  U(Uint32, head_size_v)                                                 \
  U(Uint32, chunk_size)                                                  \
  U(Uint32, chunk_base)                                                  \
  U(Uint32, chunks_in_pass)                                              \
  U(Float32, scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG

using GatedDeltaNetPrefillOutputProgram = ConfiguredProgram<GatedDeltaNetPrefillOutputProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG(F) \
  F(bool, has_decay_)                                   \
  F(bool, has_beta_)                                    \
  F(bool, qwen_gate_)                                   \
  F(bool, sigmoid_beta_)

struct GatedDeltaNetParamsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_decay, bool has_beta, bool qwen_gate, bool sigmoid_beta) {
    Config result{};
    result.has_decay_ = bool(has_decay);
    result.has_beta_ = bool(has_beta);
    result.qwen_gate_ = bool(qwen_gate);
    result.sigmoid_beta_ = bool(sigmoid_beta);

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetParams";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                        \
  U(Uint32, num_heads_v)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG

using GatedDeltaNetParamsProgram = ConfiguredProgram<GatedDeltaNetParamsProgramShader>;

#define WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG(F)

struct GatedDeltaNetCopyProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, element_count)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG

using GatedDeltaNetCopyProgram = ConfiguredProgram<GatedDeltaNetCopyProgramShader>;

#define WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG(F)

struct GatedDeltaNetUnpackQkvProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetUnpackQkv";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                            \
  U(Uint32, query_size)                                              \
  U(Uint32, value_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG

using GatedDeltaNetUnpackQkvProgram = ConfiguredProgram<GatedDeltaNetUnpackQkvProgramShader>;

#define WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_CONFIG(F)

struct GatedDeltaNetNormalizeProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetNormalize";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                           \
  U(Uint32, num_heads_q)                                            \
  U(Uint32, head_size_qk)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_NORMALIZE_PROGRAM_CONFIG

using GatedDeltaNetNormalizeProgram = ConfiguredProgram<GatedDeltaNetNormalizeProgramShader>;

#define WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_CONFIG(F) \
  F(bool, has_cu_seqlens_)

struct GatedDeltaNetChunkPrepareProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_cu_seqlens) {
    Config result{};
    result.has_cu_seqlens_ = bool(has_cu_seqlens);

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetChunkPrepare";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                               \
  U(Uint32, batch_size)                                                 \
  U(Uint32, num_heads_q)                                                \
  U(Uint32, num_heads_v)                                                \
  U(Uint32, head_size_qk)                                               \
  U(Uint32, head_size_v)                                                \
  U(Uint32, total_chunks)                                               \
  U(Uint32, chunk_elements)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_CHUNK_PREPARE_PROGRAM_CONFIG

using GatedDeltaNetChunkPrepareProgram = ConfiguredProgram<GatedDeltaNetChunkPrepareProgramShader>;

#define WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_CONFIG(F) \
  F(uint32_t, head_size_)                                   \
  F(bool, has_cu_seqlens_)                                  \
  F(bool, has_initial_state_)                               \
  F(bool, state_alias_)                                     \
  F(bool, output_final_state_)

struct GatedDeltaNetChunkScanProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t head_size, bool has_cu_seqlens, bool has_initial_state,
                           bool state_alias, bool output_final_state) {
    Config result{};
    result.head_size_ = uint32_t(head_size);
    result.has_cu_seqlens_ = bool(has_cu_seqlens);
    result.has_initial_state_ = bool(has_initial_state);
    result.state_alias_ = bool(state_alias);
    result.output_final_state_ = bool(output_final_state);

    return result;
  }
  static constexpr std::string_view name = "GatedDeltaNetChunkScan";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total_tokens)                                            \
  U(Uint32, batch_size)                                              \
  U(Uint32, num_heads_v)                                             \
  U(Uint32, head_size_qk)                                            \
  U(Uint32, head_size_v)                                             \
  U(Uint32, total_chunks)                                            \
  U(Uint32, chunk_elements)                                          \
  U(Float32, scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_DELTA_NET_CHUNK_SCAN_PROGRAM_CONFIG

using GatedDeltaNetChunkScanProgram = ConfiguredProgram<GatedDeltaNetChunkScanProgramShader>;

class GatedDeltaNet final : public WebGpuKernel {
 public:
  explicit GatedDeltaNet(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  GatedDeltaNetUpdateRule update_rule_;
  float scale_;
  int state_update_capacity_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool qk_l2_norm_;
  bool chunkwise_prefill_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
