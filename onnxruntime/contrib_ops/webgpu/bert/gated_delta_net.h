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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_PROGRAM_CONFIG);
    Config(GatedDeltaNetUpdateRule update_rule, bool has_cu_seqlens, bool has_initial_state,
           bool initial_state_in_final_state, bool output_final_state, bool qwen_gate, bool sigmoid_beta,
           bool qk_l2_norm, bool use_packed_params, bool capture_state_updates, bool vectorized_value_io)
        : update_rule_(update_rule),
          has_cu_seqlens_(has_cu_seqlens),
          has_initial_state_(has_initial_state),
          initial_state_in_final_state_(initial_state_in_final_state),
          output_final_state_(output_final_state),
          qwen_gate_(qwen_gate),
          sigmoid_beta_(sigmoid_beta),
          qk_l2_norm_(qk_l2_norm),
          use_packed_params_(use_packed_params),
          capture_state_updates_(capture_state_updates),
          vectorized_value_io_(vectorized_value_io) {}
  };
  static constexpr std::string_view name = "GatedDeltaNet";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"state_update_capacity", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_GATED_DELTA_NET_PROGRAM_CONFIG

using GatedDeltaNetProgram = ConfiguredProgram<GatedDeltaNetProgramShader>;

#define WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG(F)

struct GatedDeltaNetClearProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatedDeltaNetClear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"element_count", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_CLEAR_PROGRAM_CONFIG

using GatedDeltaNetClearProgram = ConfiguredProgram<GatedDeltaNetClearProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG(F)

struct GatedDeltaNetPrefillPrepareProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatedDeltaNetPrefillPrepare";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"chunk_base", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_PREPARE_PROGRAM_CONFIG

using GatedDeltaNetPrefillPrepareProgram = ConfiguredProgram<GatedDeltaNetPrefillPrepareProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG(F) \
  F(bool, has_initial_state_)                                 \
  F(bool, has_carry_state_)                                   \
  F(bool, output_final_state_)

struct GatedDeltaNetPrefillScanProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG);
    Config(bool has_initial_state, bool has_carry_state, bool output_final_state)
        : has_initial_state_(has_initial_state),
          has_carry_state_(has_carry_state),
          output_final_state_(output_final_state) {}
  };
  static constexpr std::string_view name = "GatedDeltaNetPrefillScan";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32},
      {"is_last_pass", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_SCAN_PROGRAM_CONFIG

using GatedDeltaNetPrefillScanProgram = ConfiguredProgram<GatedDeltaNetPrefillScanProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG(F)

struct GatedDeltaNetPrefillOutputProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatedDeltaNetPrefillOutput";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"chunk_base", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_GATED_DELTA_NET_PREFILL_OUTPUT_PROGRAM_CONFIG

using GatedDeltaNetPrefillOutputProgram = ConfiguredProgram<GatedDeltaNetPrefillOutputProgramShader>;

#define WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG(F) \
  F(bool, has_decay_)                                   \
  F(bool, has_beta_)                                    \
  F(bool, qwen_gate_)                                   \
  F(bool, sigmoid_beta_)

struct GatedDeltaNetParamsProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG);
    Config(bool has_decay, bool has_beta, bool qwen_gate, bool sigmoid_beta)
        : has_decay_(has_decay), has_beta_(has_beta), qwen_gate_(qwen_gate), sigmoid_beta_(sigmoid_beta) {}
  };
  static constexpr std::string_view name = "GatedDeltaNetParams";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_PARAMS_PROGRAM_CONFIG

using GatedDeltaNetParamsProgram = ConfiguredProgram<GatedDeltaNetParamsProgramShader>;

#define WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG(F)

struct GatedDeltaNetCopyProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatedDeltaNetCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"element_count", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_COPY_PROGRAM_CONFIG

using GatedDeltaNetCopyProgram = ConfiguredProgram<GatedDeltaNetCopyProgramShader>;

#define WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG(F)

struct GatedDeltaNetUnpackQkvProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatedDeltaNetUnpackQkv";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"query_size", ProgramUniformVariableDataType::Uint32},
      {"value_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATED_DELTA_NET_UNPACK_QKV_PROGRAM_CONFIG

using GatedDeltaNetUnpackQkvProgram = ConfiguredProgram<GatedDeltaNetUnpackQkvProgramShader>;

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
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
