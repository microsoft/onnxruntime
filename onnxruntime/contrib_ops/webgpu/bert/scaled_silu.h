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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SCALED_SI_L_U_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_scale) {
    Config result{};
    result.has_scale_ = bool(has_scale);

    return result;
  }
  static constexpr std::string_view name = "ScaledSiLU";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SCALED_SI_L_U_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, count)                                      \
  U(Float32, alpha)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SCALED_SI_L_U_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SCALED_SI_L_U_PROGRAM_SHADER_UNIFORMS
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
