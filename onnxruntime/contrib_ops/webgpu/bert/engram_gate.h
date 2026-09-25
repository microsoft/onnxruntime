// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Computes the scalar gate for each (token, g) row. The gate does not depend on the output channel,
// so one workgroup computes it once per row instead of every channel recomputing the key projection.
class EngramGateScalarProgram final : public Program<EngramGateScalarProgram> {
 public:
  EngramGateScalarProgram() : Program{"EngramGateScalar"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"embedding_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};

// Applies the per-row gate to the value projection, one invocation per output element.
class EngramGateProgram final : public Program<EngramGateProgram> {
 public:
  EngramGateProgram() : Program{"EngramGate"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"embedding_size", ProgramUniformVariableDataType::Uint32});
};

class EngramGateNormProgram final : public Program<EngramGateNormProgram> {
 public:
  EngramGateNormProgram() : Program{"EngramGateNorm"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};

class EngramGate final : public WebGpuKernel {
 public:
  explicit EngramGate(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
