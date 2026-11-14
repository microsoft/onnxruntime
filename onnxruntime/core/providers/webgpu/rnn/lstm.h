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
// to_state=true:  src [batch, H] -> dst [num_dir, batch, H] at dir offset  (for Y_h/Y_c output)
// to_state=false: src [num_dir, batch, H] at dir offset -> dst [batch, H]  (for initial state extraction)
class LstmStateCopyProgram final : public Program<LstmStateCopyProgram> {
 public:
  LstmStateCopyProgram(bool to_state, int layout, bool has_seq_lens = false)
      : Program{"LstmStateCopy"}, to_state_(to_state), layout_(layout), has_seq_lens_(has_seq_lens) {}
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

// Transposes per-direction weights [D, 4H, K] to [D, K, 4H] (the GEMM B operand), optionally casting.
class LstmPackWeightsProgram final : public Program<LstmPackWeightsProgram> {
 public:
  LstmPackWeightsProgram() : Program{"LstmPackWeights"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"rows", ProgramUniformVariableDataType::Uint32},   // 4H
      {"cols", ProgramUniformVariableDataType::Uint32});  // K
};

// Copies an ONNX state tensor (initial_h/initial_c, or Y_h/Y_c) to or from the f32 [D, B, H] state.
class LstmStateTransferProgram final : public Program<LstmStateTransferProgram> {
 public:
  LstmStateTransferProgram(bool to_output, int layout, bool has_seq_lens)
      : Program{"LstmStateTransfer"}, to_output_(to_output), layout_(layout), has_seq_lens_(has_seq_lens) {}
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

// One timestep of every direction at once, from the precomputed input projection X * W^T and the
// recurrent projection H_prev * R^T: adds the biases, applies peepholes, clip and the activations, and
// writes the new state and Y.
class LstmGatesProgram final : public Program<LstmGatesProgram> {
 public:
  LstmGatesProgram(bool has_bias, bool has_peephole, bool has_Y, bool has_seq_lens, bool input_forget,
                   bool has_clip, int layout, std::vector<std::string> activations)
      : Program{"LstmGates"},
        has_bias_(has_bias),
        has_peephole_(has_peephole),
        has_Y_(has_Y),
        has_seq_lens_(has_seq_lens),
        input_forget_(input_forget),
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
  bool has_bias_;
  bool has_peephole_;
  bool has_Y_;
  bool has_seq_lens_;
  bool input_forget_;
  bool has_clip_;
  int layout_;
  std::vector<std::string> activations_;  // WGSL function names, 3 per direction (f, g, h)
};

class Lstm final : public WebGpuKernel {
 public:
  Lstm(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;
  Status PrePackInternal(ComputeContextBase& context, const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

 private:
  // Per-timestep fallback for when the input projection does not fit in one storage binding.
  Status ComputeStepwise(ComputeContext& context) const;
  Status ComputeWithGemm(ComputeContext& context) const;

  // Transposed weights for the GEMM path, packed once when W / R are constant: W^T as [D, I, 4H] in
  // the input type, R^T as [D, H, 4H] in f32 (it multiplies the f32 recurrent state).
  std::unique_ptr<Tensor> packed_w_;
  std::unique_ptr<Tensor> packed_r_;
  mutable MatMulComputeDispatcher matmul_;

  std::string direction_;
  int64_t hidden_size_;
  float clip_;
  int64_t input_forget_;
  int64_t layout_;
  std::vector<std::string> activations_;
};

}  // namespace webgpu
}  // namespace onnxruntime
