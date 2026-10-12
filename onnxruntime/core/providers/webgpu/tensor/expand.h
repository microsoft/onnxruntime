// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_EXPAND_PROGRAM_CONFIG(F)   \
  F(bool, input_last_dim_divisible_by_4_) \
  F(bool, output_last_dim_divisible_by_4_)

struct ExpandProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_EXPAND_PROGRAM_CONFIG);
  static Config MakeConfig(const bool input_last_dim_divisible_by_4, const bool output_last_dim_divisible_by_4) {
    Config result{};
    result.input_last_dim_divisible_by_4_ = bool{input_last_dim_divisible_by_4};
    result.output_last_dim_divisible_by_4_ = bool{output_last_dim_divisible_by_4};

    return result;
  }
  static constexpr std::string_view name = "Expand";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_EXPAND_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, data_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_EXPAND_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_EXPAND_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_EXPAND_PROGRAM_CONFIG

using ExpandProgram = ConfiguredProgram<ExpandProgramShader>;

class Expand final : public WebGpuKernel {
 public:
  Expand(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Expand kernel info with appropriate type constraints based on int64 support
KernelCreateInfo CreateExpandVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateExpandKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
