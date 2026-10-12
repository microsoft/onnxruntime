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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "EngramGateScalar";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, rows)                                            \
  U(Uint32, hc_mult)                                         \
  U(Uint32, hidden_size)                                     \
  U(Uint32, hidden_vec_size)                                 \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_ENGRAM_GATE_SCALAR_PROGRAM_CONFIG

using EngramGateScalarProgram = ConfiguredProgram<EngramGateScalarProgramShader>;

// Broadcasts the per-row gate over the value channels, one invocation per vecN of output channels.
#define WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG(F)

struct EngramGateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "EngramGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_ENGRAM_GATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                    \
  U(Uint32, hc_mult)                                  \
  U(Uint32, hidden_vec_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_ENGRAM_GATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_ENGRAM_GATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_ENGRAM_GATE_PROGRAM_CONFIG

using EngramGateProgram = ConfiguredProgram<EngramGateProgramShader>;

// Applies a branchwise RMSNorm to gated_value (one hidden_size slice per hyper-connection branch)
// to produce gated_value_normed, one workgroup per (token, g) row.
#define WEBGPU_ENGRAM_GATE_NORM_PROGRAM_CONFIG(F)

struct EngramGateNormProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_ENGRAM_GATE_NORM_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "EngramGateNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_ENGRAM_GATE_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, rows)                                          \
  U(Uint32, hc_mult)                                       \
  U(Uint32, hidden_size)                                   \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_ENGRAM_GATE_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_ENGRAM_GATE_NORM_PROGRAM_SHADER_UNIFORMS
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
