// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class CompressProgram final : public Program<CompressProgram> {
 public:
  CompressProgram(bool has_axis) : Program{"Compress"}, has_axis_{has_axis} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"axis_right_stride", ProgramUniformVariableDataType::Uint32},
                                          {"compressed_dim", ProgramUniformVariableDataType::Uint32},
                                          {"input_axis_dim", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_axis_;
};

class Compress final : public WebGpuKernel {
 public:
  Compress(const OpKernelInfo& info) : WebGpuKernel(info) {
    has_axis_ = info.GetAttr<int64_t>("axis", &axis_).IsOK();
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_ = 0;
  bool has_axis_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
