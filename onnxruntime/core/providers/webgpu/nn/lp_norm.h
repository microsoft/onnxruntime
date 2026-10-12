// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_LP_NORM_PROGRAM_CONFIG(F) F(int64_t, p_)

struct LpNormProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_LP_NORM_PROGRAM_CONFIG);
  static Config MakeConfig(int64_t p) {
    Config result{};
    result.p_ = int64_t{p};

    return result;
  }
  static constexpr std::string_view name = "LpNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_LP_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, norm_count)                           \
  U(Uint32, norm_size)                            \
  U(Uint32, stride_factor)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_LP_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_LP_NORM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_LP_NORM_PROGRAM_CONFIG

using LpNormProgram = ConfiguredProgram<LpNormProgramShader>;

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
