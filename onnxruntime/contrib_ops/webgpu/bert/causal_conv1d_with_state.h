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

// Activation mode for CausalConv1DWithState
enum class CausalConv1DActivation {
  None,
  Silu,
};

CausalConv1DActivation ParseCausalConv1DActivation(const std::string& activation_str);

// Program for CausalConv1DWithState
class CausalConv1DWithStateProgram final : public Program<CausalConv1DWithStateProgram> {
 public:
  CausalConv1DWithStateProgram(CausalConv1DActivation activation, bool has_bias, bool has_conv_state,
                               int kernel_size)
      : Program{"CausalConv1DWithState"},
        activation_(activation),
        has_bias_(has_bias),
        has_conv_state_(has_conv_state),
        kernel_size_(kernel_size) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"input_length", ProgramUniformVariableDataType::Uint32},
      {"kernel_size", ProgramUniformVariableDataType::Uint32},
      {"state_length", ProgramUniformVariableDataType::Uint32},
      {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  CausalConv1DActivation activation_;
  bool has_bias_;
  bool has_conv_state_;
  [[maybe_unused]] int kernel_size_;
};

// Kernel for CausalConv1DWithState
class CausalConv1DWithState final : public WebGpuKernel {
 public:
  CausalConv1DWithState(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  CausalConv1DActivation activation_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
