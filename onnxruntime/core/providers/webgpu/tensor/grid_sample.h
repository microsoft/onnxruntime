// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// mode: 0=bilinear(linear), 1=nearest, 2=bicubic(cubic)
// padding_mode: 0=zeros, 1=border, 2=reflection

#define WEBGPU_GRID_SAMPLE_PROGRAM_CONFIG(F) \
  F(int, mode_)                              \
  F(int, padding_mode_)                      \
  F(bool, align_corners_)

struct GridSampleProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GRID_SAMPLE_PROGRAM_CONFIG);
    Config(int mode, int padding_mode, bool align_corners)
        : mode_{mode}, padding_mode_{padding_mode}, align_corners_{align_corners} {}
  };
  static constexpr std::string_view name = "GridSample";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"C", ProgramUniformVariableDataType::Uint32},
      {"H_in", ProgramUniformVariableDataType::Uint32},
      {"W_in", ProgramUniformVariableDataType::Uint32},
      {"H_out", ProgramUniformVariableDataType::Uint32},
      {"W_out", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_GRID_SAMPLE_PROGRAM_CONFIG

using GridSampleProgram = ConfiguredProgram<GridSampleProgramShader>;

class GridSample final : public WebGpuKernel {
 public:
  explicit GridSample(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int mode_{0};
  int padding_mode_{0};
  bool align_corners_{false};
};

}  // namespace webgpu
}  // namespace onnxruntime
