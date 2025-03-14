// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class CumSumProgram final : public Program<CumSumProgram> {
 public:
  CumSumProgram(int64_t exclusive, int64_t reverse) : Program{"CumSum"}, exclusive_{exclusive}, reverse_{reverse} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32,
                                           "axis", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t exclusive_;
  int64_t reverse_;
};

class GatherElements final : public WebGpuKernel {
 public:
  GatherElements(const OpKernelInfo& info) : WebGpuKernel(info) {
    exclusive_ = info.GetAttrOrDefault<int64_t>("exclusive", 0);
    reverse_ = info.GetAttrOrDefault<int64_t>("reverse", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t exclusive_;
  int64_t reverse_;
};

}  // namespace webgpu
}  // namespace onnxruntime