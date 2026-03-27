// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// Fills a GPU buffer with zeros.
class LstmZeroFillProgram final : public Program<LstmZeroFillProgram> {
 public:
  LstmZeroFillProgram() : Program{"LstmZeroFill"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32});
};

// Copies between flat [batch, H] and directional [num_dir, batch, H] (or [batch, num_dir, H]).
// to_state=true:  src [batch, H] -> dst [num_dir, batch, H] at dir offset  (for Y_h/Y_c output)
// to_state=false: src [num_dir, batch, H] at dir offset -> dst [batch, H]  (for initial state extraction)
class LstmStateCopyProgram final : public Program<LstmStateCopyProgram> {
 public:
  LstmStateCopyProgram(bool to_state, int layout)
      : Program{"LstmStateCopy"}, to_state_(to_state), layout_(layout) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32});

 private:
  bool to_state_;
  int layout_;
};

// Per-timestep LSTM cell compute.
// All h_prev/c_prev use flat [batch, H] indexing (initial state is pre-loaded into temp buffers).
// Inputs: x, w, r, h_prev, c_prev, [b], [p]
// Outputs: h_new, c_new, [y_out]
class LstmCellProgram final : public Program<LstmCellProgram> {
 public:
  LstmCellProgram(bool has_bias, bool has_peephole, bool has_Y, bool has_seq_lens,
                  bool input_forget, bool has_clip, int layout,
                  const std::string& f_activation_fn,
                  const std::string& g_activation_fn,
                  const std::string& h_activation_fn)
      : Program{"LstmCell"},
        has_bias_(has_bias),
        has_peephole_(has_peephole),
        has_Y_(has_Y),
        has_seq_lens_(has_seq_lens),
        input_forget_(input_forget),
        has_clip_(has_clip),
        layout_(layout),
        f_activation_fn_(f_activation_fn),
        g_activation_fn_(g_activation_fn),
        h_activation_fn_(h_activation_fn) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"input_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"timestep", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"clip_value", ProgramUniformVariableDataType::Float32});

 private:
  bool has_bias_;
  bool has_peephole_;
  bool has_Y_;
  bool has_seq_lens_;
  bool input_forget_;
  bool has_clip_;
  int layout_;
  std::string f_activation_fn_;
  std::string g_activation_fn_;
  std::string h_activation_fn_;
};

// Writes h_new values to the Y output tensor with optional seq_lens masking.
// Used when the cell program cannot include Y output due to storage buffer limits.
class LstmWriteYProgram final : public Program<LstmWriteYProgram> {
 public:
  LstmWriteYProgram(bool has_seq_lens, int layout)
      : Program{"LstmWriteY"}, has_seq_lens_(has_seq_lens), layout_(layout) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"timestep", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_seq_lens_;
  int layout_;
};

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
