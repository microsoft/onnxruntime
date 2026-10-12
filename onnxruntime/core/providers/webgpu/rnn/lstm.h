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
// to_state=true:  src [batch, H] -> dst [num_dir, batch, H] at dir offset  (for Y_h/Y_c output)
// to_state=false: src [num_dir, batch, H] at dir offset -> dst [batch, H]  (for initial state extraction)
#define WEBGPU_LSTM_STATE_COPY_PROGRAM_CONFIG(F) \
  F(bool, to_state_)                             \
  F(int, layout_)                                \
  F(bool, has_seq_lens_)

struct LstmStateCopyProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LSTM_STATE_COPY_PROGRAM_CONFIG);
  static Config MakeConfig(bool to_state, int layout, bool has_seq_lens = false) {
    Config result{};
    result.to_state_ = bool(to_state);
    result.layout_ = int(layout);
    result.has_seq_lens_ = bool(has_seq_lens);

    return result;
  }
  static constexpr std::string_view name = "LstmStateCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_LSTM_STATE_COPY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                   \
  U(Uint32, hidden_size)                                  \
  U(Uint32, direction)                                    \
  U(Uint32, num_directions)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LSTM_STATE_COPY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LSTM_STATE_COPY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_LSTM_STATE_COPY_PROGRAM_CONFIG

using LstmStateCopyProgram = ConfiguredProgram<LstmStateCopyProgramShader>;

// Per-timestep LSTM cell compute.
// All h_prev/c_prev use flat [batch, H] indexing (initial state is pre-loaded into temp buffers).
// Inputs: x, w, r, h_prev, c_prev, [b], [p]
// Outputs: h_new, c_new, [y_out]
#define WEBGPU_LSTM_CELL_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                       \
  F(bool, has_peephole_)                   \
  F(bool, has_Y_)                          \
  F(bool, has_seq_lens_)                   \
  F(bool, input_forget_)                   \
  F(bool, has_clip_)                       \
  F(int, layout_)                          \
  F(std::string, f_activation_fn_)         \
  F(std::string, g_activation_fn_)         \
  F(std::string, h_activation_fn_)

struct LstmCellProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LSTM_CELL_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias, bool has_peephole, bool has_Y, bool has_seq_lens, bool input_forget, bool has_clip,
                           int layout, const std::string& f_activation_fn, const std::string& g_activation_fn,
                           const std::string& h_activation_fn) {
    Config result{};
    result.has_bias_ = bool(has_bias);
    result.has_peephole_ = bool(has_peephole);
    result.has_Y_ = bool(has_Y);
    result.has_seq_lens_ = bool(has_seq_lens);
    result.input_forget_ = bool(input_forget);
    result.has_clip_ = bool(has_clip);
    result.layout_ = int(layout);
    result.f_activation_fn_ = std::string(f_activation_fn);
    result.g_activation_fn_ = std::string(g_activation_fn);
    result.h_activation_fn_ = std::string(h_activation_fn);

    return result;
  }
  static constexpr std::string_view name = "LstmCell";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_LSTM_CELL_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                             \
  U(Uint32, input_size)                             \
  U(Uint32, hidden_size)                            \
  U(Uint32, direction)                              \
  U(Uint32, num_directions)                         \
  U(Uint32, timestep)                               \
  U(Uint32, seq_length)                             \
  U(Float32, clip_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LSTM_CELL_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LSTM_CELL_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_LSTM_CELL_PROGRAM_CONFIG

using LstmCellProgram = ConfiguredProgram<LstmCellProgramShader>;

// Writes h_new values to the Y output tensor with optional seq_lens masking.
// Used when the cell program cannot include Y output due to storage buffer limits.
#define WEBGPU_LSTM_WRITE_Y_PROGRAM_CONFIG(F) \
  F(bool, has_seq_lens_)                      \
  F(int, layout_)

struct LstmWriteYProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LSTM_WRITE_Y_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_seq_lens, int layout) {
    Config result{};
    result.has_seq_lens_ = bool(has_seq_lens);
    result.layout_ = int(layout);

    return result;
  }
  static constexpr std::string_view name = "LstmWriteY";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_LSTM_WRITE_Y_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                \
  U(Uint32, hidden_size)                               \
  U(Uint32, direction)                                 \
  U(Uint32, num_directions)                            \
  U(Uint32, timestep)                                  \
  U(Uint32, seq_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LSTM_WRITE_Y_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LSTM_WRITE_Y_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_LSTM_WRITE_Y_PROGRAM_CONFIG

using LstmWriteYProgram = ConfiguredProgram<LstmWriteYProgramShader>;

class Lstm final : public WebGpuKernel {
 public:
  Lstm(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string direction_;
  int64_t hidden_size_;
  float clip_;
  int64_t input_forget_;
  int64_t layout_;
  std::vector<std::string> activations_;
};

}  // namespace webgpu
}  // namespace onnxruntime
