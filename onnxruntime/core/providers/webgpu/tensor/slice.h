// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include <iostream>

namespace onnxruntime {
namespace webgpu {

class SliceProgram final : public Program<SliceProgram> {
 public:
  SliceProgram() : Program{"Slice"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"starts", ProgramUniformVariableDataType::Uint32},
                                          {"steps", ProgramUniformVariableDataType::Uint32},
                                          {"signs", ProgramUniformVariableDataType::Int32});
};

class Slice final : public WebGpuKernel {
 public:
  Slice(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrs("starts", attr_starts_).IsOK();
    info.GetAttrs("ends", attr_ends_).IsOK();
    info.GetAttrs("axes", attr_axes_).IsOK();
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

}  // namespace webgpu
}  // namespace onnxruntime