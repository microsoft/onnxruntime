// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// Copies between flat [batch, H] and directional [num_dir, batch, H] (or [batch, num_dir, H]).
// to_state=true:  src [batch, H] -> dst [num_dir, batch, H] at dir offset  (for Y_h output)
// to_state=false: src [num_dir, batch, H] at dir offset -> dst [batch, H]  (for initial state extraction)
class GruStateCopyProgram final : public Program<GruStateCopyProgram> {
 public:
  GruStateCopyProgram(bool to_state, int layout, bool has_seq_lens = false)
      : Program{"GruStateCopy"}, to_state_(to_state), layout_(layout), has_seq_lens_(has_seq_lens) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"direction", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32});

 private:
  bool to_state_;
  int layout_;
  bool has_seq_lens_;
};

// Pass 1 of a GRU timestep: compute the update (z) and reset (r) gates for every hidden unit.
// The reset gate must be materialized for the whole [batch, H] before the hidden gate can be
// computed, because for linear_before_reset==0 the recurrent term is (r (.) H_prev) * Rh^T, which
// mixes reset values across units and therefore cannot be produced inside a single per-unit thread.
// Outputs:
//   z_out     - update gate z[j]                                          (both linear_before_reset modes)
//   reset_out - r[j] * H_prev[j] when linear_before_reset==0, else r[j]   (consumed by the hidden pass)
class GruGateProgram final : public Program<GruGateProgram> {
 public:
  GruGateProgram(bool has_bias, bool linear_before_reset, bool has_clip, int layout,
                 const std::string& f_activation_fn)
      : Program{"GruGate"},
        has_bias_(has_bias),
        linear_before_reset_(linear_before_reset),
        has_clip_(has_clip),
        layout_(layout),
        f_activation_fn_(f_activation_fn) {}

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
  bool linear_before_reset_;
  bool has_clip_;
  int layout_;
  std::string f_activation_fn_;
};

// Pass 2 of a GRU timestep: compute the hidden gate (h) and the new hidden state
//   Ht = (1 - z) (.) h + z (.) H_prev
// using the z and reset values produced by GruGateProgram.
// Inputs: x, w, r, h_prev, z, reset, [b], [seq_lens]
// Outputs: h_new, [y_out]
class GruHiddenProgram final : public Program<GruHiddenProgram> {
 public:
  GruHiddenProgram(bool has_bias, bool has_Y, bool has_seq_lens, bool linear_before_reset,
                   bool has_clip, int layout, const std::string& g_activation_fn)
      : Program{"GruHidden"},
        has_bias_(has_bias),
        has_Y_(has_Y),
        has_seq_lens_(has_seq_lens),
        linear_before_reset_(linear_before_reset),
        has_clip_(has_clip),
        layout_(layout),
        g_activation_fn_(g_activation_fn) {}

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
  bool has_Y_;
  bool has_seq_lens_;
  bool linear_before_reset_;
  bool has_clip_;
  int layout_;
  std::string g_activation_fn_;
};

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
