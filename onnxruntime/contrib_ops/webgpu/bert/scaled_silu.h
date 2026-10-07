// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_SCALED_SI_L_U_PROGRAM_CONFIG(F) F(bool, has_scale_)

struct ScaledSiLUProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SCALED_SI_L_U_PROGRAM_CONFIG);
    Config(bool has_scale) : has_scale_(has_scale) {}
  };
  static constexpr std::string_view name = "ScaledSiLU";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_SCALED_SI_L_U_PROGRAM_CONFIG

using ScaledSiLUProgram = ConfiguredProgram<ScaledSiLUProgramShader>;

class ScaledSiLU final : public WebGpuKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  float alpha_;
};

}  // namespace onnxruntime::contrib::webgpu
