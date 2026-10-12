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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_DEPTH_TO_SPACE_PROGRAM_CONFIG);
  static Config MakeConfig(int64_t* perm) {
    Config result{};
    result.perm_ = InlinedVector<int64_t>(perm, perm + 6);

    return result;
  }
  static constexpr std::string_view name = "DepthToSpace";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_DEPTH_TO_SPACE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_DEPTH_TO_SPACE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_DEPTH_TO_SPACE_PROGRAM_SHADER_UNIFORMS
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