// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GATHER_N_D_PROGRAM_CONFIG(F) \
  F(uint32_t, batch_dims_)                  \
  F(uint32_t, indices_innerest_dim_)

struct GatherNDProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GATHER_N_D_PROGRAM_CONFIG);
    Config(const uint32_t batch_dims, const uint32_t indices_innerest_dim)
        : batch_dims_{batch_dims}, indices_innerest_dim_{indices_innerest_dim} {}
  };
  static constexpr std::string_view name = "GatherND";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GATHER_N_D_PROGRAM_CONFIG

using GatherNDProgram = ConfiguredProgram<GatherNDProgramShader>;

class GatherNDBase : public WebGpuKernel {
 public:
  explicit GatherNDBase(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
    ORT_ENFORCE(batch_dims_ >= 0);
  }

 protected:
  int64_t batch_dims_;
};

class GatherND final : public GatherNDBase {
 public:
  GatherND(const OpKernelInfo& info) : GatherNDBase(info) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
