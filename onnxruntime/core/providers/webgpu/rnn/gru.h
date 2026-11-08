// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
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

// Transposes rows [row_offset, row_offset + rows) of per-direction weights [D, src_rows, K] into
// [D, K, rows] (a GEMM B operand), casting to the output type.
class GruPackWeightsProgram final : public Program<GruPackWeightsProgram> {
 public:
  GruPackWeightsProgram() : Program{"GruPackWeights"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"src_rows", ProgramUniformVariableDataType::Uint32},
      {"row_offset", ProgramUniformVariableDataType::Uint32},
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32});
};

// Copies an ONNX state tensor (initial_h, or Y_h) to or from the f32 [D, B, H] state.
class GruStateTransferProgram final : public Program<GruStateTransferProgram> {
 public:
  GruStateTransferProgram(bool to_output, int layout, bool has_seq_lens)
      : Program{"GruStateTransfer"}, to_output_(to_output), layout_(layout), has_seq_lens_(has_seq_lens) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32});

 private:
  bool to_output_;
  int layout_;
  bool has_seq_lens_;
};

// Element-wise parts of one GRU timestep without linear_before_reset, for every direction at once, given
// the precomputed input projection XW = X * W^T [D, S * B, 3H] and the recurrent GEMM results:
//   kReset:  g = H_prev * [Rz; Rr]^T [D, B, 2H]; writes z and r (.) H_prev.
//   kHidden: g = (r (.) H_prev) * Rh^T [D, B, H] plus z; writes the new state (and Y).
class GruGatesProgram final : public Program<GruGatesProgram> {
 public:
  enum class Stage { kReset,
                     kHidden };
  GruGatesProgram(Stage stage, bool has_bias, bool has_Y, bool has_seq_lens, bool has_clip, int layout,
                  std::vector<std::string> activations)
      : Program{"GruGates"},
        stage_(stage),
        has_bias_(has_bias),
        has_Y_(has_Y),
        has_seq_lens_(has_seq_lens),
        has_clip_(has_clip),
        layout_(layout),
        activations_(std::move(activations)) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"step", ProgramUniformVariableDataType::Uint32},
      {"reverse_mask", ProgramUniformVariableDataType::Uint32},  // bit d set: direction d runs backward
      {"clip_value", ProgramUniformVariableDataType::Float32});

 private:
  Stage stage_;
  bool has_bias_;
  bool has_Y_;
  bool has_seq_lens_;
  bool has_clip_;
  int layout_;
  std::vector<std::string> activations_;  // WGSL function names, 2 per direction (f, g)
};

// One whole GRU timestep with linear_before_reset for every direction: the recurrent projection
// H_prev * R^T is computed in workgroup tiles (each workgroup covers a block of hidden units for a
// block of batch rows, so R^T values are reused across rows) and fused with the gates.
class GruFusedStepProgram final : public Program<GruFusedStepProgram> {
 public:
  // `tile_units` hidden units per workgroup (a divisor of 256); each thread covers `rows_per_thread` rows.
  GruFusedStepProgram(uint32_t tile_units, uint32_t rows_per_thread, bool has_bias, bool has_Y, bool has_seq_lens,
                      bool has_clip, int layout, std::vector<std::string> activations)
      : Program{"GruFusedStep"},
        tile_units_(tile_units),
        rows_per_thread_(rows_per_thread),
        has_bias_(has_bias),
        has_Y_(has_Y),
        has_seq_lens_(has_seq_lens),
        has_clip_(has_clip),
        layout_(layout),
        activations_(std::move(activations)) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_directions", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"step", ProgramUniformVariableDataType::Uint32},
      {"reverse_mask", ProgramUniformVariableDataType::Uint32},
      {"unit_tiles", ProgramUniformVariableDataType::Uint32},
      {"batch_tiles", ProgramUniformVariableDataType::Uint32},
      {"num_tiles", ProgramUniformVariableDataType::Uint32},
      {"clip_value", ProgramUniformVariableDataType::Float32});

 private:
  uint32_t tile_units_;
  uint32_t rows_per_thread_;
  bool has_bias_;
  bool has_Y_;
  bool has_seq_lens_;
  bool has_clip_;
  int layout_;
  std::vector<std::string> activations_;  // WGSL function names, 2 per direction (f, g)
};

class Gru final : public WebGpuKernel {
 public:
  Gru(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;
  Status PrePackInternal(ComputeContextBase& context, const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

 private:
  // Per-timestep fallback for when the input projection does not fit in one storage binding.
  Status ComputeStepwise(ComputeContext& context) const;
  Status ComputeWithGemm(ComputeContext& context) const;
  Status ValidateInputs(ComputeContext& context, int64_t& seq_length, int64_t& batch_size,
                        int64_t& input_size) const;

  // f32 GEMM operands packed once when W / R are constant: W^T as [D, I, 3H]; R^T as [D, H, 3H] with
  // linear_before_reset, otherwise [Rz; Rr]^T as [D, H, 2H] and Rh^T as [D, H, H].
  std::unique_ptr<Tensor> packed_w_;
  std::unique_ptr<Tensor> packed_r_;
  std::unique_ptr<Tensor> packed_rh_;
  mutable MatMulComputeDispatcher matmul_;

  std::string direction_;
  int64_t hidden_size_;
  float clip_;
  int64_t linear_before_reset_;
  int64_t layout_;
  std::vector<std::string> activations_;
};

}  // namespace webgpu
}  // namespace onnxruntime
