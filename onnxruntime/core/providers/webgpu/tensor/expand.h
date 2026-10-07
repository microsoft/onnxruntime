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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_EXPAND_PROGRAM_CONFIG);
    Config(const bool input_last_dim_divisible_by_4, const bool output_last_dim_divisible_by_4)
        : input_last_dim_divisible_by_4_{input_last_dim_divisible_by_4},
          output_last_dim_divisible_by_4_{output_last_dim_divisible_by_4} {}
  };
  static constexpr std::string_view name = "Expand";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});
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
