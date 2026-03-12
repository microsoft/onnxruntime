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
  Linear,      // S_t = S_{t-1} + k ⊗ v
  Gated,       // S_t = exp(g) * S_{t-1} + k ⊗ v
  Delta,       // S_t = S_{t-1} + β * k ⊗ (v - S^T k)
  GatedDelta,  // S_t = exp(g) * S_{t-1} + β * k ⊗ (v - exp(g) * S^T k)
};

LinearAttentionUpdateRule ParseUpdateRule(const std::string& rule_str);

// Program for LinearAttentionRecurrent (single-token decode)
class LinearAttentionRecurrentProgram final : public Program<LinearAttentionRecurrentProgram> {
 public:
  LinearAttentionRecurrentProgram(LinearAttentionUpdateRule update_rule, bool has_decay, bool has_beta)
      : Program{"LinearAttentionRecurrent"},
        update_rule_(update_rule),
        has_decay_(has_decay),
        has_beta_(has_beta) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_dim_k", ProgramUniformVariableDataType::Uint32},
      {"head_dim_v", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  LinearAttentionUpdateRule update_rule_;
  bool has_decay_;
  bool has_beta_;
};

// Kernel for LinearAttentionRecurrent
class LinearAttentionRecurrent : public WebGpuKernel {
 public:
  LinearAttentionRecurrent(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 protected:
  LinearAttentionUpdateRule update_rule_;
  float scale_;
  int64_t chunk_size_;
};

// Program for intra-chunk attention computation
class LinearAttentionChunkIntraProgram final : public Program<LinearAttentionChunkIntraProgram> {
 public:
  LinearAttentionChunkIntraProgram(LinearAttentionUpdateRule update_rule, bool has_decay, bool has_beta)
      : Program{"LinearAttentionChunkIntra"},
        update_rule_(update_rule),
        has_decay_(has_decay),
        has_beta_(has_beta) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_dim_k", ProgramUniformVariableDataType::Uint32},
      {"head_dim_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"num_chunks", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  [[maybe_unused]] LinearAttentionUpdateRule update_rule_;
  bool has_decay_;
  bool has_beta_;
};

// Program for inter-chunk state propagation
class LinearAttentionChunkInterProgram final : public Program<LinearAttentionChunkInterProgram> {
 public:
  LinearAttentionChunkInterProgram(LinearAttentionUpdateRule update_rule, bool has_decay, bool has_beta, bool has_initial_state)
      : Program{"LinearAttentionChunkInter"},
        update_rule_(update_rule),
        has_decay_(has_decay),
        has_beta_(has_beta),
        has_initial_state_(has_initial_state) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_dim_k", ProgramUniformVariableDataType::Uint32},
      {"head_dim_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"num_chunks", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  [[maybe_unused]] LinearAttentionUpdateRule update_rule_;
  bool has_decay_;
  [[maybe_unused]] bool has_beta_;
  bool has_initial_state_;
};

// Program for full sequential computation (used for delta/gated_delta update rules)
// Delta rules have non-linear state updates (S^T k term), so chunk-parallel decomposition
// doesn't produce correct results. This program processes the full sequence sequentially.
class LinearAttentionFullSequentialProgram final : public Program<LinearAttentionFullSequentialProgram> {
 public:
  LinearAttentionFullSequentialProgram(LinearAttentionUpdateRule update_rule, bool has_decay, bool has_beta, bool has_initial_state)
      : Program{"LinearAttentionFullSequential"},
        update_rule_(update_rule),
        has_decay_(has_decay),
        has_beta_(has_beta),
        has_initial_state_(has_initial_state) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_dim_k", ProgramUniformVariableDataType::Uint32},
      {"head_dim_v", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  [[maybe_unused]] LinearAttentionUpdateRule update_rule_;
  bool has_decay_;
  bool has_beta_;
  bool has_initial_state_;
};

// Kernel for LinearAttentionChunkParallel
class LinearAttentionChunkParallel final : public LinearAttentionRecurrent {
 public:
  LinearAttentionChunkParallel(const OpKernelInfo& info);
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
