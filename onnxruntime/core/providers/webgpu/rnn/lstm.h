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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_LSTM_STATE_COPY_PROGRAM_CONFIG);
    Config(bool to_state, int layout, bool has_seq_lens = false)
        : to_state_(to_state), layout_(layout), has_seq_lens_(has_seq_lens) {}
  };
  static constexpr std::string_view name = "LstmStateCopy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32});
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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_LSTM_CELL_PROGRAM_CONFIG);
    Config(bool has_bias, bool has_peephole, bool has_Y, bool has_seq_lens, bool input_forget, bool has_clip,
           int layout, const std::string& f_activation_fn, const std::string& g_activation_fn,
           const std::string& h_activation_fn)
        : has_bias_(has_bias),
          has_peephole_(has_peephole),
          has_Y_(has_Y),
          has_seq_lens_(has_seq_lens),
          input_forget_(input_forget),
          has_clip_(has_clip),
          layout_(layout),
          f_activation_fn_(f_activation_fn),
          g_activation_fn_(g_activation_fn),
          h_activation_fn_(h_activation_fn) {}
  };
  static constexpr std::string_view name = "LstmCell";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"input_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"timestep", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"clip_value", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_LSTM_CELL_PROGRAM_CONFIG

using LstmCellProgram = ConfiguredProgram<LstmCellProgramShader>;

// Writes h_new values to the Y output tensor with optional seq_lens masking.
// Used when the cell program cannot include Y output due to storage buffer limits.
#define WEBGPU_LSTM_WRITE_Y_PROGRAM_CONFIG(F) \
  F(bool, has_seq_lens_)                      \
  F(int, layout_)

struct LstmWriteYProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_LSTM_WRITE_Y_PROGRAM_CONFIG);
    Config(bool has_seq_lens, int layout) : has_seq_lens_(has_seq_lens), layout_(layout) {}
  };
  static constexpr std::string_view name = "LstmWriteY";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"timestep", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32});
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
