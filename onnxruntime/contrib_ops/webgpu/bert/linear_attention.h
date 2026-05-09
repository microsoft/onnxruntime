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
using onnxruntime::webgpu::ComputeContext;

// Update rule enumeration
enum class LinearAttentionUpdateRule {
  Invalid,
  Linear,     // S_t = S_{t-1} + k ⊗ v
  Gated,      // S_t = exp(g) * S_{t-1} + k ⊗ v
  Delta,      // S_t = S_{t-1} + β * k ⊗ (v - S^T k)
  GatedDelta  // S_t = exp(g) * S_{t-1} + β * k ⊗ (v - exp(g) * S^T k)
};

LinearAttentionUpdateRule ParseUpdateRule(const std::string& rule_str);

// WebGPU program for the fused linear attention kernel.
// Each workgroup processes one (batch, head, dv_tile) combination.
// Threads within a workgroup (one per dk row) cooperate on reductions.
class LinearAttentionProgram final : public Program<LinearAttentionProgram> {
 public:
  LinearAttentionProgram(LinearAttentionUpdateRule update_rule, bool has_initial_state,
                         bool has_decay, bool has_beta, bool decay_broadcast_dk, int tile_v, int components)
      : Program{"LinearAttention"},
        update_rule_(update_rule),
        has_initial_state_(has_initial_state),
        has_decay_(has_decay),
        has_beta_(has_beta),
        decay_broadcast_dk_(decay_broadcast_dk),
        tile_v_(tile_v),
        components_(components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"head_dim_k", ProgramUniformVariableDataType::Uint32},
      {"head_dim_v", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"num_dv_tiles", ProgramUniformVariableDataType::Uint32},
      {"heads_per_group", ProgramUniformVariableDataType::Uint32},
      {"kv_per_k_head", ProgramUniformVariableDataType::Uint32},
      {"q_num_heads", ProgramUniformVariableDataType::Uint32},
      {"n_k_heads", ProgramUniformVariableDataType::Uint32});

 private:
  LinearAttentionUpdateRule update_rule_;
  bool has_initial_state_;
  bool has_decay_;
  bool has_beta_;
  bool decay_broadcast_dk_;
  int tile_v_;
  int components_;
};

// Kernel for LinearAttention
class LinearAttention : public WebGpuKernel {
 public:
  LinearAttention(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  float scale_;
  int q_num_heads_;
  int kv_num_heads_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
