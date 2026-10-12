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

#define WEBGPU_FAST_GELU_PROGRAM_CONFIG(F) F(int, bias_components_)

struct FastGeluProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_FAST_GELU_PROGRAM_CONFIG);
  static Config MakeConfig(int bias_components) {
    Config result{};
    result.bias_components_ = int{bias_components};

    return result;
  }
  static constexpr std::string_view name = "FastGelu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_FAST_GELU_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, vec_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_FAST_GELU_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_FAST_GELU_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_FAST_GELU_PROGRAM_CONFIG

using FastGeluProgram = ConfiguredProgram<FastGeluProgramShader>;

class FastGelu final : public WebGpuKernel {
 public:
  FastGelu(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
