// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// mode: 0=bilinear(linear), 1=nearest, 2=bicubic(cubic)
// padding_mode: 0=zeros, 1=border, 2=reflection

class GridSampleProgram final : public Program<GridSampleProgram> {
 public:
  GridSampleProgram(int mode, int padding_mode, bool align_corners)
      : Program{"GridSample"},
        mode_{mode},
        padding_mode_{padding_mode},
        align_corners_{align_corners} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"C", ProgramUniformVariableDataType::Uint32},
      {"H_in", ProgramUniformVariableDataType::Uint32},
      {"W_in", ProgramUniformVariableDataType::Uint32},
      {"H_out", ProgramUniformVariableDataType::Uint32},
      {"W_out", ProgramUniformVariableDataType::Uint32});

 private:
  int mode_;
  int padding_mode_;
  bool align_corners_;
};

class GridSample final : public WebGpuKernel {
 public:
  explicit GridSample(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int mode_{0};
  int padding_mode_{0};
  bool align_corners_{false};
};

}  // namespace webgpu
}  // namespace onnxruntime
