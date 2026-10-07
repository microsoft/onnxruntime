// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Computes the scalar gate for each (token, g) row. The gate does not depend on the output channel,
// so one workgroup reduces it once per row instead of every channel repeating the reduction.
#define WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_CONFIG(F)

struct EngramGateScalarProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "EngramGateScalar";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_vec_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_CONFIG

using EngramGateScalarProgram = ConfiguredProgram<EngramGateScalarProgramShader>;

// Broadcasts the per-row gate over the value channels, one invocation per vecN of output channels.
#define WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG(F)

struct EngramGateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "EngramGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_vec_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG

using EngramGateProgram = ConfiguredProgram<EngramGateProgramShader>;

// Applies a branchwise RMSNorm to gated_value (one hidden_size slice per hyper-connection branch)
// to produce gated_value_normed, one workgroup per (token, g) row.
#define WEBGPU_ENGRAM_GATE_NORM_PROGRAM_CONFIG(F)

struct EngramGateNormProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_ENGRAM_GATE_NORM_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "EngramGateNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_ENGRAM_GATE_NORM_PROGRAM_CONFIG

using EngramGateNormProgram = ConfiguredProgram<EngramGateNormProgramShader>;

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
