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

// Activation mode for CausalConvWithState
enum class CausalConvActivation {
  Invalid,
  None,
  Silu
};

CausalConvActivation ParseCausalConvActivation(const std::string& activation_str);

// Program for CausalConvWithState
class CausalConvWithStateProgram final : public Program<CausalConvWithStateProgram> {
 public:
  CausalConvWithStateProgram(CausalConvActivation activation, bool has_bias, bool has_conv_state,
                             bool output_present_state, bool channels_last)
      : Program{"CausalConvWithState"},
        activation_(activation),
        has_bias_(has_bias),
        has_conv_state_(has_conv_state),
        output_present_state_(output_present_state),
        channels_last_(channels_last) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"input_length", ProgramUniformVariableDataType::Uint32},
      {"kernel_size", ProgramUniformVariableDataType::Uint32},
      {"dilation", ProgramUniformVariableDataType::Uint32},
      {"state_length", ProgramUniformVariableDataType::Uint32},
      {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  CausalConvActivation activation_;
  bool has_bias_;
  bool has_conv_state_;
  bool output_present_state_;
  bool channels_last_;
};

class CausalConvUpdateStateProgram final : public Program<CausalConvUpdateStateProgram> {
 public:
  explicit CausalConvUpdateStateProgram(bool channels_last)
      : Program{"CausalConvUpdateState"}, channels_last_(channels_last) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"input_length", ProgramUniformVariableDataType::Uint32},
      {"state_length", ProgramUniformVariableDataType::Uint32},
      {"update_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool channels_last_;
};

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
