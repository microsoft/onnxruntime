// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_CONFIG(F) \
  F(bool, has_scale_)                                  \
  F(bool, shared_scale_)

struct BranchwiseRMSNormProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_CONFIG);
    Config(bool has_scale, bool shared_scale) : has_scale_(has_scale), shared_scale_(shared_scale) {}
  };
  static constexpr std::string_view name = "BranchwiseRMSNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_CONFIG

using BranchwiseRMSNormProgram = ConfiguredProgram<BranchwiseRMSNormProgramShader>;

class BranchwiseRMSNorm final : public WebGpuKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::webgpu
