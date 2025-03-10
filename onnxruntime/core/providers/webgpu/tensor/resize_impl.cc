// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/resize_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

std::string GetSafeIntegerDivision(ResizeCoordinateTransformationMode transform_coordinate) {
  // The whole part and the fractional part are calculated separately due to inaccuracy of floating
  // point division. As an example, f32(21) / f32(7) may evaluate to 2.99... instead of 3, causing an
  // offset-by-one error later in floor().
  switch (transform_coordinate) {
    case ResizeCoordinateTransformationMode::ASYMMETRIC:
      return std::string("select(f32(x_resized * length_original / length_resized) + ") +
             "f32(x_resized * length_original % length_resized) / f32(length_resized), " +
             "f32(x_resized) / x_scale, x_scale < 1.0 || floor(x_scale) != x_scale)";
      break;
    case ResizeCoordinateTransformationMode::ALIGN_CORNERS:
      return std::string("select(f32(x_resized * (length_original - 1) / (length_resized - 1)) + ") +
             "f32(x_resized * (length_original - 1) % (length_resized - 1)) / f32(length_resized - 1), " +
             "0.0, length_resized == 1)";
      break;
    default:
      ORT_THROW("The transform coordinate mode does not need to use SafeIntegerDivision");
  }
}

void TransformCoordinate(std::ostream& os, ResizeCoordinateTransformationMode transform_coordinate) {
  std::string params;
  std::string body;
  switch (transform_coordinate) {
    case ResizeCoordinateTransformationMode::HALF_PIXEL:
      params = "x_resized: u32, x_scale: f32";
      body = "(f32(x_resized) + 0.5) / x_scale - 0.5";
      break;
    case ResizeCoordinateTransformationMode::ASYMMETRIC:
      params = "x_resized: u32, x_scale: f32, length_resized: u32, length_original: u32";
      body = GetSafeIntegerDivision(transform_coordinate);
      break;
    case ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL:
      params = "x_resized: u32, x_scale: f32, length_resized: u32";
      body = "select(0.0, (f32(x_resized) + 0.5) / x_scale - 0.5, length_resized > 1)";
      break;
    case ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN:
      params = "x_resized: u32, x_scale: f32";
      body = "(f32(x_resized) + 0.5) / x_scale";
      break;
    case ResizeCoordinateTransformationMode::ALIGN_CORNERS:
      params = "x_resized: u32, length_resized: u32, length_original: u32";
      body = GetSafeIntegerDivision(transform_coordinate);
      break;
    case ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE:
      params = "x_resized: u32, length_resized: u32, length_original: u32, roi_start: f32, roi_end: f32";
      body = std::string("select(0.5 * (roi_start + roi_end) * f32(length_original - 1),") +
             "roi_start * f32(length_original - 1) + (f32(x_resized) * (roi_end - roi_start) * " +
             "f32(length_original - 1)) / f32(length_resized - 1), length_resized > 1)";
      break;
    case ResizeCoordinateTransformationMode::HALF_PIXEL_SYMMETRIC:
      params = "x_resized: u32, x_scale: f32, length_resized: u32, length_original: u32";
      body = std::string("(f32(length_original) / 2.0) * (1.0 - f32(length_resized) / ") +
             "(x_scale * f32(length_original))) + (f32(x_resized) + 0.5) / x_scale - 0.5";
      break;
    default:
      ORT_THROW("unknown ResizeCoordinateTransformationMode");
  }

  os << "fn transform_coordinate(" << params << ") -> f32 {\n";
  os << "return " << body << ";\n}\n";
}

std::string GetCoordinateCaller(ResizeCoordinateTransformationMode transform_coordinate, int32_t rank) {
  std::string scales_index_str = GetElementAt("uniforms.scales", "axis", rank);
  std::string input_shape_index_str = GetElementAt("uniforms.input_shape", "axis", rank);
  std::string output_shape_index_str = GetElementAt("uniforms.output_shape", "axis", rank);
  std::string roi_start_str = GetElementAt("uniforms.roi", "axis", rank * 2);
  std::string roi_end_str = GetElementAt("uniforms.roi", "axis + " + std::to_string(rank), rank * 2);
  std::stringstream caller_ss;
  caller_ss << "transform_coordinate(output_coord, ";
  switch (transform_coordinate) {
    case ResizeCoordinateTransformationMode::HALF_PIXEL:
    case ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN:
      caller_ss << scales_index_str;
      break;
    case ResizeCoordinateTransformationMode::ASYMMETRIC:
      caller_ss << scales_index_str << ", " << output_shape_index_str << ", " << input_shape_index_str;
      break;
    case ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL:
      caller_ss << scales_index_str << ", " << output_shape_index_str;
      break;
    case ResizeCoordinateTransformationMode::ALIGN_CORNERS:
      caller_ss << output_shape_index_str << ", " << input_shape_index_str;
      break;
    case ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE:
      caller_ss << output_shape_index_str << ", " << input_shape_index_str
                << ", f32(" << roi_start_str << "), f32(" << roi_end_str << ")";
      break;
    case ResizeCoordinateTransformationMode::HALF_PIXEL_SYMMETRIC:
      caller_ss << scales_index_str << ", " << output_shape_index_str << ", " << input_shape_index_str;
      break;
    default:
      ORT_THROW("unknown ResizeCoordinateTransformationMode");
  }
  caller_ss << ")";

  return caller_ss.str();
}

void CalcNearestPixel(std::ostream& os, ResizeNearestMode mode) {
  std::string params = "x_original: f32";
  std::string body;
  switch (mode) {
    case ResizeNearestMode::SIMPLE:
      params += ", is_down_sampling: bool";
      body = "select(i32(x_original), i32(ceil(x_original)), is_down_sampling)";
      break;
    case ResizeNearestMode::ROUND_PREFER_FLOOR:
      body = "select(i32(round(x_original)), i32(floor(x_original)), x_original == f32(i32(x_original)) + 0.5)";
      break;
    case ResizeNearestMode::ROUND_PREFER_CEIL:
      body = "i32(round(x_original))";
      break;
    case ResizeNearestMode::FLOOR:
      body = "i32(floor(x_original))";
      break;
    case ResizeNearestMode::CEIL:
      body = "i32(ceil(x_original))";
      break;
    default:
      ORT_THROW("unknown ResizeNearestMode");
  }

  os << "fn calc_nearest_pixel(" << params << ") -> i32 {\n";
  os << "return " << body << ";\n}\n";
}

std::string GetNearestPixelCaller(ResizeNearestMode mode) {
  switch (mode) {
    case ResizeNearestMode::SIMPLE:
      return "calc_nearest_pixel(input_coord, uniforms.scales[axis] < 1.0)";
      break;
    case ResizeNearestMode::ROUND_PREFER_FLOOR:
    case ResizeNearestMode::ROUND_PREFER_CEIL:
    case ResizeNearestMode::FLOOR:
    case ResizeNearestMode::CEIL:
      return "calc_nearest_pixel(input_coord)";
      break;
    default:
      ORT_THROW("unknown ResizeNearestMode");
  }
}

Status ResizeNearestProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform |
                                                                   ShaderUsage::UseShapeAndStride |
                                                                   ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                                      ShaderUsage::UseShapeAndStride);

  std::string scales_index_str = GetElementAt("uniforms.scales", "axis", rank_);
  std::string input_shape_index_str = GetElementAt("uniforms.input_shape", "axis", rank_);
  std::string input_stride_index_str = GetElementAt("uniforms.input_stride", "axis", rank_ - 1);

  std::stringstream extrapolation_ss;
  if (extrapolation_enabled_) {
    extrapolation_ss << "      if ((input_coord < 0.0 || input_coord > f32(" << input_shape_index_str << " - 1))) {\n"
                     << "        " << output.SetByOffset("global_idx", "input_value_t(uniforms.extrapolation_value)") << ";\n"
                     << "        return;\n"
                     << "      }\n";
  }

  TransformCoordinate(shader.AdditionalImplementation(), coordinate_transform_mode_);
  CalcNearestPixel(shader.AdditionalImplementation(), nearest_mode_);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var input_index = u32(0);\n"
                            << "  for(var axis = 0; axis < " << rank_ << "; axis++) {\n"
                            << "    var output_coord = output_indices[axis];\n"
                            << "    if (" << scales_index_str << " != 1.0) {\n"
                            << "      let input_coord = " << GetCoordinateCaller(coordinate_transform_mode_, rank_) << ";\n"
                            << extrapolation_ss.str()
                            << "      var nearest_coord = " << GetNearestPixelCaller(nearest_mode_) << ";\n"
                            << "      if (nearest_coord >= i32(" << input_shape_index_str << ")) {\n"
                            << "        output_coord = " << input_shape_index_str << " - 1;\n"
                            << "      } else if (nearest_coord < 0) {\n"
                            << "        output_coord = 0;\n"
                            << "      } else {\n"
                            << "        output_coord = u32(nearest_coord);\n"
                            << "      }"
                            << "    }\n"
                            << "    input_index += select(output_coord * " << input_stride_index_str << ", output_coord, axis == " << rank_ - 1 << ");\n"
                            << "  }\n"
                            << "  " << output.SetByOffset("global_idx", input.GetByOffset("input_index")) << ";\n";

  return Status::OK();
}

Status ResizeNearestImpl(ComputeContext& context,
                         const Tensor* input_tensor,
                         int32_t rank,
                         gsl::span<const int64_t>& output_dims,
                         gsl::span<const float> roi,
                         gsl::span<const float> scales,
                         bool extrapolation_enabled,
                         const float extrapolation_value,
                         onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
                         onnxruntime::ResizeNearestMode nearest_mode) {
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow<uint32_t>(output_shape.Size());

  ResizeNearestProgram program{coordinate_transform_mode, nearest_mode, extrapolation_enabled, rank};
  program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(extrapolation_enabled)),
                 std::to_string(static_cast<int>(coordinate_transform_mode)),
                 std::to_string(static_cast<int>(nearest_mode)))
      .AddUniformVariables({{roi}, {scales}, {output_size}, {extrapolation_value}});

  return context.RunProgram(program);
}

Status ResizeBilinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform |
                                                                   ShaderUsage::UseShapeAndStride |
                                                                   ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                                      ShaderUsage::UseShapeAndStride);

  std::string scales_index_str = GetElementAt("uniforms.scales", "axis", rank_);
  std::string input_shape_index_str = GetElementAt("uniforms.input_shape", "axis", rank_);

  std::stringstream extrapolation_ss;
  if (extrapolation_enabled_) {
    extrapolation_ss << "  if ((input_coord < 0.0 || input_coord > input_max_coord)) {\n"
                     << "    " << output.SetByOffset("global_idx", "input_value_t(uniforms.extrapolation_value)") << ";\n"
                     << "    return;\n"
                     << "  }\n";
  }

  TransformCoordinate(shader.AdditionalImplementation(), coordinate_transform_mode_);
  std::string transform_coordinate_caller = GetCoordinateCaller(coordinate_transform_mode_, rank_);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var input_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var axis = " << rank_ - 2 << ";\n"
                            << "  let input_height = u32(" << input_shape_index_str << ");\n"
                            << "  var input_max_coord = f32(input_height - 1);\n"
                            << "  var output_coord = output_indices[axis];\n"
                            << "  var input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_ss.str()
                            << "  input_coord = max(0.0, min(input_coord, input_max_coord));\n"
                            << "  let input_y_coord_int = u32(input_coord);\n"
                            << "  let y_weight_0 = select(input_coord - f32(input_y_coord_int), 0.5, input_coord >= input_max_coord);\n"
                            << "  axis = " << rank_ - 1 << ";\n"
                            << "  let input_width = u32(" << input_shape_index_str << ");\n"
                            << "  input_max_coord = f32(input_width - 1);\n"
                            << "  output_coord = output_indices[axis];\n"
                            << "  input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_ss.str()
                            << "  input_coord = max(0.0, min(input_coord, input_max_coord));\n"
                            << "  let input_x_coord_int = u32(input_coord);\n"
                            << "  let x_weight_0 = select(input_coord - f32(input_x_coord_int), 0.5, input_coord >= input_max_coord);\n"
                            << "  let end_of_h = (input_y_coord_int >= input_height - 1);\n"
                            << "  let end_of_w = (input_x_coord_int >= input_width - 1);\n"
                            << "  let rank = " << rank_ << ";\n"
                            << "  input_indices[rank - 2] = input_y_coord_int;\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  let x00 = " << input.GetByIndices("input_indices") << ";\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x10 = select(" << input.GetByIndices("input_indices") << ", x00, end_of_w);\n"
                            << "  input_indices[rank - 2] = input_y_coord_int + 1;\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  let x01 = select(" << input.GetByIndices("input_indices") << ", x00, end_of_h);\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x11 = select(select(" << input.GetByIndices("input_indices") << ", x10, end_of_h), x01, end_of_w);\n"
                            << "  let y_weight_1 = 1.0 - y_weight_0;\n"
                            << "  let x_weight_1 = 1.0 - x_weight_0;\n"
                            << "  var value = input_value_t(f32(x00) * y_weight_1 * x_weight_1 + f32(x01) * y_weight_0 * x_weight_1 + f32(x10) * "
                            << "y_weight_1 * x_weight_0 + f32(x11) * y_weight_0 * x_weight_0);\n"
                            << "  " << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status ResizeBilinearImpl(ComputeContext& context,
                          const Tensor* input_tensor,
                          int32_t rank,
                          gsl::span<const int64_t>& output_dims,
                          gsl::span<const float> roi,
                          gsl::span<const float> scales,
                          bool extrapolation_enabled,
                          const float extrapolation_value,
                          onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode) {
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow<uint32_t>(output_shape.Size());

  ResizeBilinearProgram program{coordinate_transform_mode, extrapolation_enabled, rank};
  program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(extrapolation_enabled)),
                 std::to_string(static_cast<int>(coordinate_transform_mode)))
      .AddUniformVariables({{roi}, {scales}, {output_size}, {extrapolation_value}});

  return context.RunProgram(program);
}

Status ResizeTrilinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform |
                                                                   ShaderUsage::UseShapeAndStride |
                                                                   ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                                      ShaderUsage::UseShapeAndStride);

  std::string scales_index_str = GetElementAt("uniforms.scales", "axis", rank_);
  std::string input_shape_index_str = GetElementAt("uniforms.input_shape", "axis", rank_);

  std::stringstream extrapolation_ss;
  if (extrapolation_enabled_) {
    extrapolation_ss << "  if ((input_coord < 0.0 || input_coord > f32(" << input_shape_index_str << " - 1))) {\n"
                     << "    " << output.SetByOffset("global_idx", "input_value_t(uniforms.extrapolation_value)") << ";\n"
                     << "    return;\n"
                     << "  }\n";
  }

  TransformCoordinate(shader.AdditionalImplementation(), coordinate_transform_mode_);
  std::string transform_coordinate_caller = GetCoordinateCaller(coordinate_transform_mode_, rank_);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var input_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var axis = " << rank_ - 3 << ";\n"
                            << "  let input_depth = u32(" << input_shape_index_str << ");\n"
                            << "  var input_max_coord = f32(input_depth - 1);\n"
                            << "  var output_coord = output_indices[axis];\n"
                            << "  var input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_ss.str()
                            << "  input_coord = max(0.0, min(input_coord, input_max_coord));\n"
                            << "  let input_z_coord_int = u32(input_coord);\n"
                            << "  let z_weight_0 = select(input_coord - f32(input_z_coord_int), 0.5, input_coord >= input_max_coord);\n"
                            << "  axis = " << rank_ - 2 << ";\n"
                            << "  let input_height = u32(" << input_shape_index_str << ");\n"
                            << "  input_max_coord = f32(input_height - 1);\n"
                            << "  output_coord = output_indices[axis];\n"
                            << "  input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_ss.str()
                            << "  input_coord = max(0.0, min(input_coord, input_max_coord));\n"
                            << "  let input_y_coord_int = u32(input_coord);\n"
                            << "  let y_weight_0 = select(input_coord - f32(input_y_coord_int), 0.5, input_coord >= input_max_coord);\n"
                            << "  axis = " << rank_ - 1 << ";\n"
                            << "  let input_width = u32(" << input_shape_index_str << ");\n"
                            << "  input_max_coord = f32(input_width - 1);\n"
                            << "  output_coord = output_indices[axis];\n"
                            << "  input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_ss.str()
                            << "  input_coord = max(0.0, min(input_coord, input_max_coord));\n"
                            << "  let input_x_coord_int = u32(input_coord);\n"
                            << "  let x_weight_0 = select(input_coord - f32(input_x_coord_int), 0.5, input_coord >= input_max_coord);\n"
                            << "  let end_of_d = (input_z_coord_int >= input_depth - 1);\n"
                            << "  let end_of_h = (input_y_coord_int >= input_height - 1);\n"
                            << "  let end_of_w = (input_x_coord_int >= input_width - 1);\n"
                            << "  let rank = " << rank_ << ";\n"
                            << "  input_indices[rank - 3] = input_z_coord_int;\n"
                            << "  input_indices[rank - 2] = input_y_coord_int;\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  let x000 = " << input.GetByIndices("input_indices") << ";\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x100 = select(" << input.GetByIndices("input_indices") << ", x000, end_of_w);\n"
                            << "  input_indices[rank - 2] = input_y_coord_int + 1;\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  let x010 = select(" << input.GetByIndices("input_indices") << ", x000, end_of_h);\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x110 = select(select(" << input.GetByIndices("input_indices") << ", x100, end_of_h), x010, end_of_w);\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  input_indices[rank - 2] = input_y_coord_int;\n"
                            << "  input_indices[rank - 3] = select(input_z_coord_int + 1, input_z_coord_int, end_of_d);\n"
                            << "  let x001 = select(" << input.GetByIndices("input_indices") << ", x000, end_of_d);\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x101 = select(" << input.GetByIndices("input_indices") << ", x001, end_of_w);\n"
                            << "  input_indices[rank - 2] = input_y_coord_int + 1;\n"
                            << "  input_indices[rank - 1] = input_x_coord_int;\n"
                            << "  let x011 = select(" << input.GetByIndices("input_indices") << ", x001, end_of_h);\n"
                            << "  input_indices[rank - 1] = input_x_coord_int + 1;\n"
                            << "  let x111 = select(select(" << input.GetByIndices("input_indices") << ", x101, end_of_h), x011, end_of_w);\n"
                            << "  let z_weight_1 = 1.0 - z_weight_0;\n"
                            << "  let y_weight_1 = 1.0 - y_weight_0;\n"
                            << "  let x_weight_1 = 1.0 - x_weight_0;\n"
                            << "  var value = input_value_t("
                            << "f32(x000) * z_weight_1 * y_weight_1 * x_weight_1 + f32(x010) * z_weight_1 * y_weight_0 * x_weight_1 + "
                            << "f32(x100) * z_weight_1 * y_weight_1 * x_weight_0 + f32(x110) * z_weight_1 * y_weight_0 * x_weight_0 + "
                            << "f32(x001) * z_weight_0 * y_weight_1 * x_weight_1 + f32(x011) * z_weight_0 * y_weight_0 * x_weight_1 + "
                            << "f32(x101) * z_weight_0 * y_weight_1 * x_weight_0 + f32(x111) * z_weight_0 * y_weight_0 * x_weight_0"
                            << ");\n"
                            << "  " << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status ResizeTrilinearImpl(ComputeContext& context,
                           const Tensor* input_tensor,
                           int32_t rank,
                           gsl::span<const int64_t>& output_dims,
                           gsl::span<const float> roi,
                           gsl::span<const float> scales,
                           bool extrapolation_enabled,
                           const float extrapolation_value,
                           onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode) {
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow<uint32_t>(output_shape.Size());

  ResizeTrilinearProgram program{coordinate_transform_mode, extrapolation_enabled, rank};
  program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(extrapolation_enabled)),
                 std::to_string(static_cast<int>(coordinate_transform_mode)))
      .AddUniformVariables({{roi}, {scales}, {output_size}, {extrapolation_value}});

  return context.RunProgram(program);
}

Status ResizeBiCubicProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform |
                                                                   ShaderUsage::UseShapeAndStride |
                                                                   ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                                      ShaderUsage::UseShapeAndStride);

  std::string scales_index_str = GetElementAt("uniforms.scales", "axis", rank_);
  std::string input_shape_index_str = GetElementAt("uniforms.input_shape", "axis", rank_);

  std::stringstream exclude_outside_ss;
  std::stringstream extrapolation_enabled_ss;
  if (exclude_outside_) {
    exclude_outside_ss << "  coeff[0] = select(coeff[0], 0.0, (input_coord_int - 1 < 0 || input_coord_int - 1 >= input_max_coord));\n"
                       << "  coeff[1] = select(coeff[1], 0.0, (input_coord_int + 0 < 0 || input_coord_int + 0 >= input_max_coord));\n"
                       << "  coeff[2] = select(coeff[2], 0.0, (input_coord_int + 1 < 0 || input_coord_int + 1 >= input_max_coord));\n"
                       << "  coeff[3] = select(coeff[3], 0.0, (input_coord_int + 2 < 0 || input_coord_int + 2 >= input_max_coord));\n"
                       << "  coeff_sum = dot(coeff, vec4<f32>(1.0));\n";
  }

  if (extrapolation_enabled_) {
    extrapolation_enabled_ss << "  if ((input_coord < 0.0 || input_coord > f32(input_max_coord - 1))) {\n"
                             << "    " << output.SetByOffset("global_idx", "input_value_t(uniforms.extrapolation_value)") << ";\n"
                             << "    return;\n"
                             << "  }\n";
  }

  std::stringstream coeff_ss;
  coeff_ss << "  coeff[0] = ((cubic_coeff_a * (s_coord + 1.0) - 5.0 * cubic_coeff_a) * (s_coord + 1.0) + 8.0 * cubic_coeff_a) * (s_coord + 1.0) - 4.0 * cubic_coeff_a;\n"
           << "  coeff[1] = ((cubic_coeff_a + 2.0) * s_coord - (cubic_coeff_a + 3.0)) * s_coord * s_coord + 1.0;\n"
           << "  coeff[2] = ((cubic_coeff_a + 2.0) * (1.0 - s_coord) - (cubic_coeff_a + 3.0)) * (1.0 - s_coord) * (1.0 - s_coord) + 1.0;\n"
           << "  coeff[3] = ((cubic_coeff_a * (2.0 - s_coord) - 5.0 * cubic_coeff_a) * (2.0 - s_coord) + 8.0 * cubic_coeff_a) * (2.0 - s_coord) - 4.0 * cubic_coeff_a;\n";

  std::stringstream cubic_interpolation_rowwise_ss;
  cubic_interpolation_rowwise_ss << "  input_indices[" << rank_ - 2 << "] = u32(clamp(y, 0, input_height - 1));\n"
                                 << "  input_indices[" << rank_ - 1 << "] = u32(clamp(input_x_coord_int - 1, 0, input_width - 1));\n"
                                 << "  value_rowwise = x_coeff[0] * " << input.GetByIndices("input_indices") << ";\n"
                                 << "  input_indices[" << rank_ - 1 << "] = u32(clamp(input_x_coord_int, 0, input_width - 1));\n"
                                 << "  value_rowwise += x_coeff[1] * " << input.GetByIndices("input_indices") << ";\n"
                                 << "  input_indices[" << rank_ - 1 << "] = u32(clamp(input_x_coord_int + 1, 0, input_width - 1));\n"
                                 << "  value_rowwise += x_coeff[2] * " << input.GetByIndices("input_indices") << ";\n"
                                 << "  input_indices[" << rank_ - 1 << "] = u32(clamp(input_x_coord_int + 2, 0, input_width - 1));\n"
                                 << "  value_rowwise += x_coeff[3] * " << input.GetByIndices("input_indices") << ";\n";

  TransformCoordinate(shader.AdditionalImplementation(), coordinate_transform_mode_);
  std::string transform_coordinate_caller = GetCoordinateCaller(coordinate_transform_mode_, rank_);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var input_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var axis = " << rank_ - 2 << ";\n"
                            << "  let input_height = i32(" << input_shape_index_str << ");\n"
                            << "  var input_max_coord = input_height;\n"
                            << "  var output_coord = output_indices[axis];\n"
                            << "  var input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_enabled_ss.str()
                            << "  var coeff_sum = 1.0;\n"
                            << "  let cubic_coeff_a = uniforms.cubic_coeff_a;\n"
                            << "  var input_coord_int = i32(floor(input_coord));\n"
                            << "  var s_coord = abs(input_coord - f32(input_coord_int));\n"
                            << "  var coeff = vec4<f32>(0.0);\n"
                            << coeff_ss.str()
                            << exclude_outside_ss.str()
                            << "  let input_y_coord_int = input_coord_int;\n"
                            << "  let y_coeff = coeff / coeff_sum;\n"
                            << "  axis = " << rank_ - 1 << ";\n"
                            << "  let input_width = i32(" << input_shape_index_str << ");\n"
                            << "  input_max_coord = input_width;\n"
                            << "  output_coord = output_indices[axis];\n"
                            << "  input_coord = select(f32(output_coord), " << transform_coordinate_caller << " , " << scales_index_str << " != 1.0);\n"
                            << extrapolation_enabled_ss.str()
                            << "  input_coord_int = i32(floor(input_coord));\n"
                            << "  s_coord = abs(input_coord - f32(input_coord_int));\n"
                            << coeff_ss.str()
                            << exclude_outside_ss.str()
                            << "  let input_x_coord_int = input_coord_int;\n"
                            << "  let x_coeff = coeff / coeff_sum;\n"
                            << "  var y = input_y_coord_int - 1;\n"
                            << "  var value_rowwise = 0.0;\n"
                            << "  var value = 0.0;\n"
                            << cubic_interpolation_rowwise_ss.str()
                            << "  value += y_coeff[0] * value_rowwise;\n"
                            << "  y = input_y_coord_int;\n"
                            << cubic_interpolation_rowwise_ss.str()
                            << "  value += y_coeff[1] * value_rowwise;\n"
                            << "  y = input_y_coord_int + 1;\n"
                            << cubic_interpolation_rowwise_ss.str()
                            << "  value += y_coeff[2] * value_rowwise;\n"
                            << "  y = input_y_coord_int + 2;\n"
                            << cubic_interpolation_rowwise_ss.str()
                            << "  value += y_coeff[3] * value_rowwise;\n"
                            << output.SetByOffset("global_idx", "input_value_t(value)");

  return Status::OK();
}

Status ResizeBiCubicImpl(ComputeContext& context,
                         const Tensor* input_tensor,
                         int32_t rank,
                         gsl::span<const int64_t>& output_dims,
                         gsl::span<const float> roi,
                         gsl::span<const float> scales,
                         bool extrapolation_enabled,
                         const float extrapolation_value,
                         float cubic_coeff_a,
                         bool exclude_outside,
                         onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode) {
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow<uint32_t>(output_shape.Size());

  ResizeBiCubicProgram program{coordinate_transform_mode, extrapolation_enabled, exclude_outside, rank};
  program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(extrapolation_enabled)),
                 std::to_string(static_cast<int>(exclude_outside)),
                 std::to_string(static_cast<int>(coordinate_transform_mode)))
      .AddUniformVariables({{roi}, {scales}, {output_size}, {extrapolation_value}, {cubic_coeff_a}});

  return context.RunProgram(program);
}

Status ResizeImpl(ComputeContext& context,
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
                  onnxruntime::ResizeNearestMode nearest_mode) {
  int32_t rank = static_cast<int32_t>(output_dims.size());
  // We support a special case of bilinear or bicubic if the input data is 4D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  bool is_2D = (rank == 2 || rank == 4);

  // We support a special case of trilinear or tricubic if the input data is 5D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  bool is_3D = (rank == 3 || rank == 5);

  // Should not hit this as we have already validated input rank/scales and we provide verbose error messages
  // to the user.
  ORT_ENFORCE(is_2D || is_3D, "Only bilinear/trilinear and bicubic modes are supported in Resize");

  switch (upsample_mode) {
    case UpsampleMode::NN:
      return ResizeNearestImpl(context, input, rank, output_dims, roi, scales, extrapolation_enabled,
                               extrapolation_value, coordinate_transform_mode, nearest_mode);
      break;
    case UpsampleMode::LINEAR:
      if (is_2D) {
        return ResizeBilinearImpl(context, input, rank, output_dims, roi, scales, extrapolation_enabled,
                                  extrapolation_value, coordinate_transform_mode);
      } else if (is_3D) {
        return ResizeTrilinearImpl(context, input, rank, output_dims, roi, scales, extrapolation_enabled,
                                   extrapolation_value, coordinate_transform_mode);
      }
      ORT_THROW("Resize support 2-D and 3-D dimensions in LINEAR mode.");
      break;
    case UpsampleMode::CUBIC:
      if (is_2D) {
        return ResizeBiCubicImpl(context, input, rank, output_dims, roi, scales, extrapolation_enabled,
                                 extrapolation_value, cubic_coeff_a, exclude_outside, coordinate_transform_mode);
      }
      ORT_THROW("Resize supports only 2-D in CUBIC mode.");
    default:
      ORT_THROW("Only nearest, bilinear/trilinear and bicubic modes are supported in Resize");
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
