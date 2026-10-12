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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GRID_SAMPLE_PROGRAM_CONFIG);
  static Config MakeConfig(int mode, int padding_mode, bool align_corners) {
    Config result{};
    result.mode_ = int{mode};
    result.padding_mode_ = int{padding_mode};
    result.align_corners_ = bool{align_corners};

    return result;
  }
  static constexpr std::string_view name = "GridSample";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GRID_SAMPLE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                              \
  U(Uint32, C)                                        \
  U(Uint32, H_in)                                     \
  U(Uint32, W_in)                                     \
  U(Uint32, H_out)                                    \
  U(Uint32, W_out)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GRID_SAMPLE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GRID_SAMPLE_PROGRAM_SHADER_UNIFORMS
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
