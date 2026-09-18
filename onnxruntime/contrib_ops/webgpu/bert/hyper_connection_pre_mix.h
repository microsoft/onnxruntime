// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class HyperConnectionPreMixProgram final : public Program<HyperConnectionPreMixProgram> {
 public:
  explicit HyperConnectionPreMixProgram(int gate_layout)
      : Program{"HyperConnectionPreMix"}, gate_layout_(gate_layout) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"reduction_scale", ProgramUniformVariableDataType::Float32});

 private:
  int gate_layout_;
};

class HyperConnectionPreMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

}  // namespace onnxruntime::contrib::webgpu
