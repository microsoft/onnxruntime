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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_RANGE_PROGRAM_CONFIG);
  static Config MakeConfig(int32_t data_type = 0) {
    Config result{};
    result.data_type_ = int32_t{data_type};

    return result;
  }
  static constexpr std::string_view name = "Range";

  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_RANGE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                        \
  U(Uint32, start)                              \
  U(Uint32, delta)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_RANGE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_RANGE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_RANGE_PROGRAM_CONFIG

using RangeProgram = ConfiguredProgram<RangeProgramShader>;

// Register Range kernels with conditional int64 support
void RegisterRangeKernels(KernelRegistry& kernel_registry, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
