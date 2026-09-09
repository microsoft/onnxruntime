// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/program.h"
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

class GatedDeltaNetProgram final : public Program<GatedDeltaNetProgram> {
 public:
  GatedDeltaNetProgram(GatedDeltaNetUpdateRule update_rule, bool has_cu_seqlens, bool has_initial_state,
                       bool initial_state_in_final_state, bool qwen_gate, bool sigmoid_beta, bool qk_l2_norm)
      : Program{"GatedDeltaNet"},
        update_rule_(update_rule),
        has_cu_seqlens_(has_cu_seqlens),
        has_initial_state_(has_initial_state),
        initial_state_in_final_state_(initial_state_in_final_state),
        qwen_gate_(qwen_gate),
        sigmoid_beta_(sigmoid_beta),
        qk_l2_norm_(qk_l2_norm) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  GatedDeltaNetUpdateRule update_rule_;
  bool has_cu_seqlens_;
  bool has_initial_state_;
  bool initial_state_in_final_state_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool qk_l2_norm_;
};

class GatedDeltaNet final : public WebGpuKernel {
 public:
  explicit GatedDeltaNet(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  GatedDeltaNetUpdateRule update_rule_;
  float scale_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool qk_l2_norm_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
