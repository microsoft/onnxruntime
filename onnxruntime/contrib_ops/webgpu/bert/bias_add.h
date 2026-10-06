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

#define WEBGPU_BIAS_ADD_CONFIG(F)
WEBGPU_DECLARE_CONFIG(BiasAddConfig, WEBGPU_BIAS_ADD_CONFIG);
#undef WEBGPU_BIAS_ADD_CONFIG

struct BiasAddShader {
  using Config = BiasAddConfig;
  static constexpr std::string_view name = "BiasAdd";
  static Status GenerateShaderCode(const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"channels", ProgramUniformVariableDataType::Uint32});
};

using BiasAddProgram = ConfiguredProgram<BiasAddShader>;

class BiasAdd final : public WebGpuKernel {
 public:
  BiasAdd(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
