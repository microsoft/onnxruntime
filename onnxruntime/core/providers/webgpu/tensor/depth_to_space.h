// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_DEPTH_TO_SPACE_PROGRAM_CONFIG(F) F(InlinedVector<int64_t>, perm_)

struct DepthToSpaceProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DEPTH_TO_SPACE_PROGRAM_CONFIG);
    Config(int64_t* perm) : perm_(perm, perm + 6) {}
  };
  static constexpr std::string_view name = "DepthToSpace";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_DEPTH_TO_SPACE_PROGRAM_CONFIG

using DepthToSpaceProgram = ConfiguredProgram<DepthToSpaceProgramShader>;

template <bool is_nhwc>
class DepthToSpace final : public WebGpuKernel {
 public:
  DepthToSpace(const OpKernelInfo& info) : WebGpuKernel(info) {
    blocksize_ = info.GetAttr<int64_t>("blocksize");
    std::string mode = info.GetAttrOrDefault<std::string>("mode", "DCR");
    is_dcr_ = (mode == "DCR");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t blocksize_;
  bool is_dcr_;
};

}  // namespace webgpu
}  // namespace onnxruntime