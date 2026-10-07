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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_BIAS_SPLIT_GELU_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "BiasSplitGelu";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"channels", ProgramUniformVariableDataType::Uint32});
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