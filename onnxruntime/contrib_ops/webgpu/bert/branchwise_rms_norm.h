// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class BranchwiseRMSNormProgram final : public Program<BranchwiseRMSNormProgram> {
 public:
  BranchwiseRMSNormProgram(bool has_scale, bool shared_scale)
      : Program{"BranchwiseRMSNorm"}, has_scale_(has_scale), shared_scale_(shared_scale) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_scale_;
  bool shared_scale_;
};

class BranchwiseRMSNorm final : public WebGpuKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::webgpu
