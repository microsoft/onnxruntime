// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/configured_program.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
class Range : public WebGpuKernel {
 public:
  explicit Range(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

#define WEBGPU_RANGE_PROGRAM_CONFIG(F) F(int32_t, data_type_)

struct RangeProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_RANGE_PROGRAM_CONFIG);
    Config(int32_t data_type = 0) : data_type_{data_type} {}
  };
  static constexpr std::string_view name = "Range";

  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"start", ProgramUniformVariableDataType::Uint32},
                                          {"delta", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_RANGE_PROGRAM_CONFIG

using RangeProgram = ConfiguredProgram<RangeProgramShader>;

// Register Range kernels with conditional int64 support
void RegisterRangeKernels(KernelRegistry& kernel_registry, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
