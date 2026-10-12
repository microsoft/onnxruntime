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

#define WEBGPU_BIAS_SPLIT_GELU_PROGRAM_CONFIG(F)

struct BiasSplitGeluProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_BIAS_SPLIT_GELU_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "BiasSplitGelu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_BIAS_SPLIT_GELU_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                  \
  U(Uint32, channels)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_BIAS_SPLIT_GELU_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_BIAS_SPLIT_GELU_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_BIAS_SPLIT_GELU_PROGRAM_CONFIG

using BiasSplitGeluProgram = ConfiguredProgram<BiasSplitGeluProgramShader>;

class BiasSplitGelu final : public WebGpuKernel {
 public:
  BiasSplitGelu(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime