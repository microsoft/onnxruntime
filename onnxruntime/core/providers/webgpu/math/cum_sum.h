// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CUM_SUM_PROGRAM_CONFIG(F)

struct CumSumProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CUM_SUM_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "CumSum";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_CUM_SUM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                          \
  U(Uint32, axis)                                 \
  U(Uint32, exclusive)                            \
  U(Uint32, reverse)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CUM_SUM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CUM_SUM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CUM_SUM_PROGRAM_CONFIG

using CumSumProgram = ConfiguredProgram<CumSumProgramShader>;

class CumSum final : public WebGpuKernel {
 public:
  CumSum(const OpKernelInfo& info) : WebGpuKernel(info) {
    exclusive_ = info.GetAttrOrDefault<int64_t>("exclusive", 0);
    reverse_ = info.GetAttrOrDefault<int64_t>("reverse", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t exclusive_;
  int64_t reverse_;
};

}  // namespace webgpu
}  // namespace onnxruntime