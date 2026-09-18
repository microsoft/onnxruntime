// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class ScaledSiLUProgram final : public Program<ScaledSiLUProgram> {
 public:
  explicit ScaledSiLUProgram(bool has_scale)
      : Program{"ScaledSiLU"}, has_scale_(has_scale) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});

 private:
  bool has_scale_;
};

class ScaledSiLU final : public WebGpuKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float alpha_;
};

}  // namespace onnxruntime::contrib::webgpu
