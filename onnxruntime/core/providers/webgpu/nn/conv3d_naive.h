// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class Conv3DNaiveProgram final : public Program<Conv3DNaiveProgram> {
 public:
  Conv3DNaiveProgram(const Activation& activation, bool has_bias, bool is_channels_last)
      : Program("Conv3DNaive"), activation_(activation), has_bias_(has_bias), is_channels_last_(is_channels_last) {
  }
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"filter_dims", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"x_spatial", ProgramUniformVariableDataType::Uint32},
      {"x_channels", ProgramUniformVariableDataType::Uint32});

 private:
  const Activation& activation_;
  bool has_bias_;
  bool is_channels_last_;
};

}  // namespace webgpu
}  // namespace onnxruntime
