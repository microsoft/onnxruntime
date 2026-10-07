// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_RESIZE_NEAREST_PROGRAM_CONFIG(F)                                  \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(onnxruntime::ResizeNearestMode, nearest_mode_)                               \
  F(bool, extrapolation_enabled_)                                                \
  F(int32_t, rank_)

struct ResizeNearestProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_RESIZE_NEAREST_PROGRAM_CONFIG);
    Config(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
           onnxruntime::ResizeNearestMode nearest_mode, bool extrapolation_enabled, int32_t rank)
        : coordinate_transform_mode_{coordinate_transform_mode},
          nearest_mode_{nearest_mode},
          extrapolation_enabled_{extrapolation_enabled},
          rank_{rank} {}
  };
  static constexpr std::string_view name = "ResizeNearest2D";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_RESIZE_NEAREST_PROGRAM_CONFIG

using ResizeNearestProgram = ConfiguredProgram<ResizeNearestProgramShader>;

#define WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG(F)                                 \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(int32_t, rank_)

struct ResizeBilinearProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG);
    Config(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
           int32_t rank)
        : coordinate_transform_mode_{coordinate_transform_mode},
          extrapolation_enabled_{extrapolation_enabled},
          rank_{rank} {}
  };
  static constexpr std::string_view name = "ResizeBilinear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG

using ResizeBilinearProgram = ConfiguredProgram<ResizeBilinearProgramShader>;

#define WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG(F)                                \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(int32_t, rank_)

struct ResizeTrilinearProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG);
    Config(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
           int32_t rank)
        : coordinate_transform_mode_{coordinate_transform_mode},
          extrapolation_enabled_{extrapolation_enabled},
          rank_{rank} {}
  };
  static constexpr std::string_view name = "ResizeTrilinear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG

using ResizeTrilinearProgram = ConfiguredProgram<ResizeTrilinearProgramShader>;

#define WEBGPU_RESIZE_BI_CUBIC_PROGRAM_CONFIG(F)                                 \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(bool, exclude_outside_)                                                      \
  F(int32_t, rank_)

struct ResizeBiCubicProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_RESIZE_BI_CUBIC_PROGRAM_CONFIG);
    Config(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
           bool exclude_outside, int32_t rank)
        : coordinate_transform_mode_{coordinate_transform_mode},
          extrapolation_enabled_{extrapolation_enabled},
          exclude_outside_{exclude_outside},
          rank_{rank} {}
  };
  static constexpr std::string_view name = "ResizeBiCubic";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32},
                                          {"cubic_coeff_a", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_RESIZE_BI_CUBIC_PROGRAM_CONFIG

using ResizeBiCubicProgram = ConfiguredProgram<ResizeBiCubicProgramShader>;

Status ResizeImpl(
    ComputeContext& context,
    const Tensor* input,
    const onnxruntime::UpsampleMode upsample_mode,
    gsl::span<const int64_t>& output_dims,
    gsl::span<const float> roi,
    gsl::span<const float> scales,
    bool extrapolation_enabled,
    const float extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
    onnxruntime::ResizeNearestMode nearest_mode);

}  // namespace webgpu
}  // namespace onnxruntime
