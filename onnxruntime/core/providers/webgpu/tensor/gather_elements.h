// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GATHER_ELEMENTS_PROGRAM_CONFIG(F)

struct GatherElementsProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATHER_ELEMENTS_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "GatherElements";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"axis_dim_limit", ProgramUniformVariableDataType::Int32},
                                          {"axis", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_GATHER_ELEMENTS_PROGRAM_CONFIG

using GatherElementsProgram = ConfiguredProgram<GatherElementsProgramShader>;

class GatherElements final : public WebGpuKernel {
 public:
  GatherElements(const OpKernelInfo& info) : WebGpuKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
};

}  // namespace webgpu
}  // namespace onnxruntime