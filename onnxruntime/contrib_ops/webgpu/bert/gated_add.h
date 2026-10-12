// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// output = X + Y * gate, with gate broadcast across the last dimension.
#define WEBGPU_GATED_ADD_PROGRAM_CONFIG(F)

struct GatedAddProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATED_ADD_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "GatedAdd";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_GATED_ADD_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                            \
  U(Uint32, hidden_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATED_ADD_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATED_ADD_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATED_ADD_PROGRAM_CONFIG

using GatedAddProgram = ConfiguredProgram<GatedAddProgramShader>;

class GatedAdd final : public WebGpuKernel {
 public:
  GatedAdd(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
