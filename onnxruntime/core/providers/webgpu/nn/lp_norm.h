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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_LP_NORM_PROGRAM_CONFIG);
    Config(int64_t p) : p_{p} {}
  };
  static constexpr std::string_view name = "LpNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"norm_count", ProgramUniformVariableDataType::Uint32},
      {"norm_size", ProgramUniformVariableDataType::Uint32},
      {"stride_factor", ProgramUniformVariableDataType::Uint32});
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
