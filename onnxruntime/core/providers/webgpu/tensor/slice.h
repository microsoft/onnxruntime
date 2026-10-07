// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_SLICE_PROGRAM_CONFIG(F)

struct SliceProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SLICE_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "Slice";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"starts", ProgramUniformVariableDataType::Uint32},
                                          {"steps", ProgramUniformVariableDataType::Uint32},
                                          {"signs", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_SLICE_PROGRAM_CONFIG

using SliceProgram = ConfiguredProgram<SliceProgramShader>;

class Slice final : public WebGpuKernel {
 public:
  Slice(const OpKernelInfo& info) : WebGpuKernel(info) {
    // since only opset1-9 provides these as attributes, we can safely ignore the return value
    // we handle failure in fetching the attribute in ComputeInternal
    (void)info.GetAttrs("starts", attr_starts_);
    (void)info.GetAttrs("ends", attr_ends_);
    (void)info.GetAttrs("axes", attr_axes_);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

}  // namespace webgpu
}  // namespace onnxruntime