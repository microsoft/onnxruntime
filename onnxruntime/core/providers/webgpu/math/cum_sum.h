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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_CUM_SUM_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "CumSum";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"axis", ProgramUniformVariableDataType::Uint32},
                                          {"exclusive", ProgramUniformVariableDataType::Uint32},
                                          {"reverse", ProgramUniformVariableDataType::Uint32});
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