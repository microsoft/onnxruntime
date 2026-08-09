// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace webgpu {

class ResizeNearestProgram final : public Program<ResizeNearestProgram> {
 public:
  ResizeNearestProgram(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                       onnxruntime::ResizeNearestMode nearest_mode,
                       bool extrapolation_enabled,
                       int32_t rank) : Program{"ResizeNearest2D"},
                                       coordinate_transform_mode_{coordinate_transform_mode},
                                       nearest_mode_{nearest_mode},
                                       extrapolation_enabled_{extrapolation_enabled},
                                       rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});

 private:
  onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode_;
  onnxruntime::ResizeNearestMode nearest_mode_;
  bool extrapolation_enabled_;
  int32_t rank_;
};

class ResizeBilinearProgram final : public Program<ResizeBilinearProgram> {
 public:
  ResizeBilinearProgram(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                        bool extrapolation_enabled,
                        int32_t rank) : Program{"ResizeBilinear"},
                                        coordinate_transform_mode_{coordinate_transform_mode},
                                        extrapolation_enabled_{extrapolation_enabled},
                                        rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});

 private:
  onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode_;
  bool extrapolation_enabled_;
  int32_t rank_;
};

class ResizeTrilinearProgram final : public Program<ResizeTrilinearProgram> {
 public:
  ResizeTrilinearProgram(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                         bool extrapolation_enabled,
                         int32_t rank) : Program{"ResizeTrilinear"},
                                         coordinate_transform_mode_{coordinate_transform_mode},
                                         extrapolation_enabled_{extrapolation_enabled},
                                         rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32});

 private:
  onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode_;
  bool extrapolation_enabled_;
  int32_t rank_;
};

class ResizeBiCubicProgram final : public Program<ResizeBiCubicProgram> {
 public:
  ResizeBiCubicProgram(onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                       bool extrapolation_enabled,
                       bool exclude_outside,
                       int32_t rank) : Program{"ResizeBiCubic"},
                                       coordinate_transform_mode_{coordinate_transform_mode},
                                       extrapolation_enabled_{extrapolation_enabled},
                                       exclude_outside_{exclude_outside},
                                       rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"roi", ProgramUniformVariableDataType::Float32},
                                          {"scales", ProgramUniformVariableDataType::Float32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"extrapolation_value", ProgramUniformVariableDataType::Float32},
                                          {"cubic_coeff_a", ProgramUniformVariableDataType::Float32});

 private:
  onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode_;
  bool extrapolation_enabled_;
  bool exclude_outside_;
  int32_t rank_;
};

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
