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

// Activation mode for CausalConvWithState
enum class CausalConvActivation {
  Invalid,
  None,
  Silu
};

CausalConvActivation ParseCausalConvActivation(const std::string& activation_str);

// Program for CausalConvWithState
#define WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG(F) \
  F(CausalConvActivation, activation_)                  \
  F(bool, has_bias_)                                    \
  F(bool, has_conv_state_)                              \
  F(bool, output_present_state_)                        \
  F(bool, channels_last_)

struct CausalConvWithStateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG);
  static Config MakeConfig(CausalConvActivation activation, bool has_bias, bool has_conv_state, bool output_present_state,
                           bool channels_last) {
    Config result{};
    result.activation_ = CausalConvActivation(activation);
    result.has_bias_ = bool(has_bias);
    result.has_conv_state_ = bool(has_conv_state);
    result.output_present_state_ = bool(output_present_state);
    result.channels_last_ = bool(channels_last);

    return result;
  }
  static constexpr std::string_view name = "CausalConvWithState";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                          \
  U(Uint32, channels)                                            \
  U(Uint32, input_length)                                        \
  U(Uint32, kernel_size)                                         \
  U(Uint32, dilation)                                            \
  U(Uint32, state_length)                                        \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG

using CausalConvWithStateProgram = ConfiguredProgram<CausalConvWithStateProgramShader>;

#define WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_CONFIG(F) F(bool, channels_last_)

struct CausalConvUpdateStateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_CONFIG);
  static Config MakeConfig(bool channels_last) {
    Config result{};
    result.channels_last_ = bool(channels_last);

    return result;
  }
  static constexpr std::string_view name = "CausalConvUpdateState";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, channels)                                              \
  U(Uint32, input_length)                                          \
  U(Uint32, state_length)                                          \
  U(Uint32, update_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_CONFIG

using CausalConvUpdateStateProgram = ConfiguredProgram<CausalConvUpdateStateProgramShader>;

// Kernel for CausalConvWithState
class CausalConvWithState final : public WebGpuKernel {
 public:
  CausalConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  CausalConvActivation activation_;
  int64_t ndim_;
  int dilation_;
  bool channels_last_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
