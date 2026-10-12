// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// Copies between flat [batch, H] and directional [num_dir, batch, H] (or [batch, num_dir, H]).
// to_state=true:  src [batch, H] -> dst [num_dir, batch, H] at dir offset  (for Y_h output)
// to_state=false: src [num_dir, batch, H] at dir offset -> dst [batch, H]  (for initial state extraction)
#define WEBGPU_GRU_STATE_COPY_PROGRAM_CONFIG(F) \
  F(bool, to_state_)                            \
  F(int, layout_)                               \
  F(bool, has_seq_lens_)

struct GruStateCopyProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GRU_STATE_COPY_PROGRAM_CONFIG);
  static Config MakeConfig(bool to_state, int layout, bool has_seq_lens = false) {
    Config result{};
    result.to_state_ = bool(to_state);
    result.layout_ = int(layout);
    result.has_seq_lens_ = bool(has_seq_lens);

    return result;
  }
  static constexpr std::string_view name = "GruStateCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_GRU_STATE_COPY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                  \
  U(Uint32, hidden_size)                                 \
  U(Uint32, direction)                                   \
  U(Uint32, num_directions)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GRU_STATE_COPY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GRU_STATE_COPY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GRU_STATE_COPY_PROGRAM_CONFIG

using GruStateCopyProgram = ConfiguredProgram<GruStateCopyProgramShader>;

// Pass 1 of a GRU timestep: compute the update (z) and reset (r) gates for every hidden unit.
// The reset gate must be materialized for the whole [batch, H] before the hidden gate can be
// computed, because for linear_before_reset==0 the recurrent term is (r (.) H_prev) * Rh^T, which
// mixes reset values across units and therefore cannot be produced inside a single per-unit thread.
// Outputs:
//   z_out     - update gate z[j]                                          (both linear_before_reset modes)
//   reset_out - r[j] * H_prev[j] when linear_before_reset==0, else r[j]   (consumed by the hidden pass)
#define WEBGPU_GRU_GATE_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                      \
  F(bool, linear_before_reset_)           \
  F(bool, has_clip_)                      \
  F(int, layout_)                         \
  F(std::string, f_activation_fn_)

struct GruGateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GRU_GATE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias, bool linear_before_reset, bool has_clip, int layout, const std::string& f_activation_fn) {
    Config result{};
    result.has_bias_ = bool(has_bias);
    result.linear_before_reset_ = bool(linear_before_reset);
    result.has_clip_ = bool(has_clip);
    result.layout_ = int(layout);
    result.f_activation_fn_ = std::string(f_activation_fn);

    return result;
  }
  static constexpr std::string_view name = "GruGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GRU_GATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                            \
  U(Uint32, input_size)                            \
  U(Uint32, hidden_size)                           \
  U(Uint32, direction)                             \
  U(Uint32, num_directions)                        \
  U(Uint32, timestep)                              \
  U(Uint32, seq_length)                            \
  U(Float32, clip_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GRU_GATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GRU_GATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GRU_GATE_PROGRAM_CONFIG

using GruGateProgram = ConfiguredProgram<GruGateProgramShader>;

// Pass 2 of a GRU timestep: compute the hidden gate (h) and the new hidden state
//   Ht = (1 - z) (.) h + z (.) H_prev
// using the z and reset values produced by GruGateProgram.
// Inputs: x, w, r, h_prev, z, reset, [b], [seq_lens]
// Outputs: h_new, [y_out]
#define WEBGPU_GRU_HIDDEN_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                        \
  F(bool, has_Y_)                           \
  F(bool, has_seq_lens_)                    \
  F(bool, linear_before_reset_)             \
  F(bool, has_clip_)                        \
  F(int, layout_)                           \
  F(std::string, g_activation_fn_)

struct GruHiddenProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GRU_HIDDEN_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias, bool has_Y, bool has_seq_lens, bool linear_before_reset, bool has_clip, int layout,
                           const std::string& g_activation_fn) {
    Config result{};
    result.has_bias_ = bool(has_bias);
    result.has_Y_ = bool(has_Y);
    result.has_seq_lens_ = bool(has_seq_lens);
    result.linear_before_reset_ = bool(linear_before_reset);
    result.has_clip_ = bool(has_clip);
    result.layout_ = int(layout);
    result.g_activation_fn_ = std::string(g_activation_fn);

    return result;
  }
  static constexpr std::string_view name = "GruHidden";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GRU_HIDDEN_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                              \
  U(Uint32, input_size)                              \
  U(Uint32, hidden_size)                             \
  U(Uint32, direction)                               \
  U(Uint32, num_directions)                          \
  U(Uint32, timestep)                                \
  U(Uint32, seq_length)                              \
  U(Float32, clip_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GRU_HIDDEN_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GRU_HIDDEN_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GRU_HIDDEN_PROGRAM_CONFIG

using GruHiddenProgram = ConfiguredProgram<GruHiddenProgramShader>;

class Gru final : public WebGpuKernel {
 public:
  Gru(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string direction_;
  int64_t hidden_size_;
  float clip_;
  int64_t linear_before_reset_;
  int64_t layout_;
  std::vector<std::string> activations_;
};

}  // namespace webgpu
}  // namespace onnxruntime
