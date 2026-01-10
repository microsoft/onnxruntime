// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class GatherElementsProgram final : public Program<GatherElementsProgram> {
 public:
  GatherElementsProgram() : Program{"GatherElements"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"axis_dim_limit", ProgramUniformVariableDataType::Int32},
                                          {"axis", ProgramUniformVariableDataType::Int32});
};

class GatherElements final : public WebGpuKernel {
 public:
  GatherElements(const OpKernelInfo& info) : WebGpuKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
};

}  // namespace webgpu
}  // namespace onnxruntime