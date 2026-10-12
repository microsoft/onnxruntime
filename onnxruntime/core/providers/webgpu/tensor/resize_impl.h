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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_RESIZE_NEAREST_PROGRAM_CONFIG);
  static Config MakeConfig(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                           onnxruntime::ResizeNearestMode nearest_mode, bool extrapolation_enabled, int32_t rank) {
    Config result{};
    result.coordinate_transform_mode_ = onnxruntime::ResizeCoordinateTransformationMode{coordinate_transform_mode};
    result.nearest_mode_ = onnxruntime::ResizeNearestMode{nearest_mode};
    result.extrapolation_enabled_ = bool{extrapolation_enabled};
    result.rank_ = int32_t{rank};

    return result;
  }
  static constexpr std::string_view name = "ResizeNearest2D";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_RESIZE_NEAREST_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, roi)                                        \
  U(Float32, scales)                                     \
  U(Uint32, output_size)                                 \
  U(Float32, extrapolation_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_RESIZE_NEAREST_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_RESIZE_NEAREST_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_RESIZE_NEAREST_PROGRAM_CONFIG

using ResizeNearestProgram = ConfiguredProgram<ResizeNearestProgramShader>;

#define WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG(F)                                 \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(int32_t, rank_)

struct ResizeBilinearProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG);
  static Config MakeConfig(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
                           int32_t rank) {
    Config result{};
    result.coordinate_transform_mode_ = onnxruntime::ResizeCoordinateTransformationMode{coordinate_transform_mode};
    result.extrapolation_enabled_ = bool{extrapolation_enabled};
    result.rank_ = int32_t{rank};

    return result;
  }
  static constexpr std::string_view name = "ResizeBilinear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_RESIZE_BILINEAR_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, roi)                                         \
  U(Float32, scales)                                      \
  U(Uint32, output_size)                                  \
  U(Float32, extrapolation_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_RESIZE_BILINEAR_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_RESIZE_BILINEAR_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_RESIZE_BILINEAR_PROGRAM_CONFIG

using ResizeBilinearProgram = ConfiguredProgram<ResizeBilinearProgramShader>;

#define WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG(F)                                \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(int32_t, rank_)

struct ResizeTrilinearProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG);
  static Config MakeConfig(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
                           int32_t rank) {
    Config result{};
    result.coordinate_transform_mode_ = onnxruntime::ResizeCoordinateTransformationMode{coordinate_transform_mode};
    result.extrapolation_enabled_ = bool{extrapolation_enabled};
    result.rank_ = int32_t{rank};

    return result;
  }
  static constexpr std::string_view name = "ResizeTrilinear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_RESIZE_TRILINEAR_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, roi)                                          \
  U(Float32, scales)                                       \
  U(Uint32, output_size)                                   \
  U(Float32, extrapolation_value)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_RESIZE_TRILINEAR_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_RESIZE_TRILINEAR_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_RESIZE_TRILINEAR_PROGRAM_CONFIG

using ResizeTrilinearProgram = ConfiguredProgram<ResizeTrilinearProgramShader>;

#define WEBGPU_RESIZE_BI_CUBIC_PROGRAM_CONFIG(F)                                 \
  F(onnxruntime::ResizeCoordinateTransformationMode, coordinate_transform_mode_) \
  F(bool, extrapolation_enabled_)                                                \
  F(bool, exclude_outside_)                                                      \
  F(int32_t, rank_)

struct ResizeBiCubicProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_RESIZE_BI_CUBIC_PROGRAM_CONFIG);
  static Config MakeConfig(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode, bool extrapolation_enabled,
                           bool exclude_outside, int32_t rank) {
    Config result{};
    result.coordinate_transform_mode_ = onnxruntime::ResizeCoordinateTransformationMode{coordinate_transform_mode};
    result.extrapolation_enabled_ = bool{extrapolation_enabled};
    result.exclude_outside_ = bool{exclude_outside};
    result.rank_ = int32_t{rank};

    return result;
  }
  static constexpr std::string_view name = "ResizeBiCubic";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_RESIZE_BI_CUBIC_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, roi)                                         \
  U(Float32, scales)                                      \
  U(Uint32, output_size)                                  \
  U(Float32, extrapolation_value)                         \
  U(Float32, cubic_coeff_a)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_RESIZE_BI_CUBIC_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_RESIZE_BI_CUBIC_PROGRAM_SHADER_UNIFORMS
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
