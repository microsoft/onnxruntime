// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/bert/causal_conv_with_state.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Program for VarlenCausalConvWithState. One shader invocation owns a single
// (request, channel) pair and walks every token of its request serially, so
// request boundaries and per-channel carry state stay isolated.
#define WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                                           \
  F(bool, has_state_)                                          \
  F(bool, state_in_final_state_)                               \
  F(bool, has_state_update_)                                   \
  F(bool, has_capture_count_)                                  \
  F(bool, use_silu_)

struct VarlenCausalConvWithStateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_bias, bool has_state, bool state_in_final_state, bool has_state_update, bool has_capture_count,
                           bool use_silu) {
    Config result{};
    result.has_bias_ = bool(has_bias);
    result.has_state_ = bool(has_state);
    result.state_in_final_state_ = bool(state_in_final_state);
    result.has_state_update_ = bool(has_state_update);
    result.has_capture_count_ = bool(has_capture_count);
    result.use_silu_ = bool(use_silu);

    return result;
  }
  static constexpr std::string_view name = "VarlenCausalConvWithState";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, channels)                                                   \
  U(Uint32, kernel_size)                                                \
  U(Uint32, dilation)                                                   \
  U(Uint32, pad)                                                        \
  U(Uint32, state_update_capacity)                                      \
  U(Uint32, batch_size)                                                 \
  U(Uint32, total_tokens)                                               \
  U(Uint32, num_invocations)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG

using VarlenCausalConvWithStateProgram = ConfiguredProgram<VarlenCausalConvWithStateProgramShader>;

// Kernel for VarlenCausalConvWithState (packed token-major, variable-length batches).
class VarlenCausalConvWithState final : public WebGpuKernel {
 public:
  VarlenCausalConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  CausalConvActivation activation_;
  int dilation_;
  int state_update_capacity_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
