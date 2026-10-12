// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
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
#define WEBGPU_LINEAR_ATTENTION_PROGRAM_CONFIG(F) \
  F(LinearAttentionUpdateRule, update_rule_)      \
  F(bool, has_initial_state_)                     \
  F(bool, initial_state_in_present_state_)        \
  F(bool, has_decay_)                             \
  F(bool, has_beta_)                              \
  F(bool, decay_broadcast_dk_)                    \
  F(int, tile_v_)                                 \
  F(int, components_)                             \
  F(int, subgroup_min_size_)

struct LinearAttentionProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LINEAR_ATTENTION_PROGRAM_CONFIG);
  static Config MakeConfig(LinearAttentionUpdateRule update_rule, bool has_initial_state, bool initial_state_in_present_state,
                           bool has_decay, bool has_beta, bool decay_broadcast_dk, int tile_v, int components, int subgroup_min_size) {
    Config result{};
    result.update_rule_ = LinearAttentionUpdateRule(update_rule);
    result.has_initial_state_ = bool(has_initial_state);
    result.initial_state_in_present_state_ = bool(initial_state_in_present_state);
    result.has_decay_ = bool(has_decay);
    result.has_beta_ = bool(has_beta);
    result.decay_broadcast_dk_ = bool(decay_broadcast_dk);
    result.tile_v_ = int(tile_v);
    result.components_ = int(components);
    result.subgroup_min_size_ = int(subgroup_min_size);

    return result;
  }
  static constexpr std::string_view name = "LinearAttention";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_LINEAR_ATTENTION_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                    \
  U(Uint32, num_heads)                                     \
  U(Uint32, seq_length)                                    \
  U(Uint32, head_dim_k)                                    \
  U(Uint32, head_dim_v)                                    \
  U(Float32, scale)                                        \
  U(Uint32, num_dv_tiles)                                  \
  U(Uint32, heads_per_group)                               \
  U(Uint32, kv_per_k_head)                                 \
  U(Uint32, q_num_heads)                                   \
  U(Uint32, n_k_heads)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LINEAR_ATTENTION_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LINEAR_ATTENTION_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_LINEAR_ATTENTION_PROGRAM_CONFIG

using LinearAttentionProgram = ConfiguredProgram<LinearAttentionProgramShader>;

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
