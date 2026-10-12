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
WEBGPU_DECLARE_SPECIALIZATION(BiasAddConfig, WEBGPU_BIAS_ADD_CONFIG);
#undef WEBGPU_BIAS_ADD_CONFIG

struct BiasAddShader {
  using Config = BiasAddConfig;
  static constexpr std::string_view name = "BiasAdd";
  static Status GenerateShaderCode(const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_BIAS_ADD_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                   \
  U(Uint32, channels)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_BIAS_ADD_SHADER_UNIFORMS);
#undef WEBGPU_BIAS_ADD_SHADER_UNIFORMS
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
