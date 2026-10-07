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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG);
    Config(CausalConvActivation activation, bool has_bias, bool has_conv_state, bool output_present_state,
           bool channels_last)
        : activation_(activation),
          has_bias_(has_bias),
          has_conv_state_(has_conv_state),
          output_present_state_(output_present_state),
          channels_last_(channels_last) {}
  };
  static constexpr std::string_view name = "CausalConvWithState";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"input_length", ProgramUniformVariableDataType::Uint32},
      {"kernel_size", ProgramUniformVariableDataType::Uint32},
      {"dilation", ProgramUniformVariableDataType::Uint32},
      {"state_length", ProgramUniformVariableDataType::Uint32},
      {"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_CAUSAL_CONV_WITH_STATE_PROGRAM_CONFIG

using CausalConvWithStateProgram = ConfiguredProgram<CausalConvWithStateProgramShader>;

#define WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_CONFIG(F) F(bool, channels_last_)

struct CausalConvUpdateStateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_CAUSAL_CONV_UPDATE_STATE_PROGRAM_CONFIG);
    Config(bool channels_last) : channels_last_(channels_last) {}
  };
  static constexpr std::string_view name = "CausalConvUpdateState";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"input_length", ProgramUniformVariableDataType::Uint32},
      {"state_length", ProgramUniformVariableDataType::Uint32},
      {"update_size", ProgramUniformVariableDataType::Uint32});
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
