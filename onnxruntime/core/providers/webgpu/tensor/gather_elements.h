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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATHER_ELEMENTS_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatherElements";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GATHER_ELEMENTS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                  \
  U(Int32, axis_dim_limit)                                \
  U(Int32, axis)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATHER_ELEMENTS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATHER_ELEMENTS_PROGRAM_SHADER_UNIFORMS
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