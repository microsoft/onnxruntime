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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_FAST_GELU_PROGRAM_CONFIG);
    Config(int bias_components) : bias_components_{bias_components} {}
  };
  static constexpr std::string_view name = "FastGelu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});
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
