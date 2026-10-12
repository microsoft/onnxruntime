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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_scale, bool shared_scale) {
    Config result{};
    result.has_scale_ = bool(has_scale);
    result.shared_scale_ = bool(shared_scale);

    return result;
  }
  static constexpr std::string_view name = "BranchwiseRMSNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, hidden)                                             \
  U(Uint32, branches)                                           \
  U(Uint32, groups)                                             \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_BRANCHWISE_R_M_S_NORM_PROGRAM_SHADER_UNIFORMS
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
