// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class LpNormProgram final : public Program<LpNormProgram> {
 public:
  LpNormProgram(int64_t p) : Program{"LpNorm"}, p_{p} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"norm_count", ProgramUniformVariableDataType::Uint32},
      {"norm_size", ProgramUniformVariableDataType::Uint32},
      {"stride_factor", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t p_;
};

class LpNorm final : public WebGpuKernel {
 public:
  LpNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<int64_t>("p", &p_, 2);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int64_t p_;
};

}  // namespace webgpu
}  // namespace onnxruntime
