// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include <string>
#include <sstream>
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/nn/conv_backprop_webgpu.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

Status ConvTranspose2DProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& w = shader.AddInput("W", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& dy = shader.AddInput("dy", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  std::vector<const ShaderVariableHelper*> inputs = {&w, &dy};
  if (has_bias_) {
    const auto& bias = shader.AddInput("bias");
    inputs.push_back(&bias);
  }
  auto row_dim = is_channels_last_ ? 1 : 2;
  auto col_dim = is_channels_last_ ? 2 : 3;
  auto channel_dim = is_channels_last_ ? 3 : 1;
  auto calculate_result = [&]() -> std::string {
    std::stringstream ss;
    if (pack_input_as4_) {
      if (a_components_ == 4) {
        ss << "let xValue = " << dy.GetByOffset("x_offset") << ";\n"
           << "let wValue = " << w.GetByOffset("w_offset") << ";\n"
           << "dotProd = dotProd + dot(xValue, wValue);\n"
           << "x_offset += 1;\n"
           << "w_offset += 1;\n";
      } else if (a_components_ == 2) {
        ss << "let xValue = vec4<f32>(" << dy.GetByOffset("x_offset") << ", " << dy.GetByOffset("x_offset + 1") << ");\n"
           << "let wValue = vec4<f32>(" << w.GetByOffset("w_offset") << ", " << w.GetByOffset("w_offset + 1u") << ");\n"
           << "dotProd = dotProd + dot(xValue, wValue);\n"
           << "x_offset += 2;\n"
           << "w_offset += 2;\n";
      } else if (a_components_ == 1) {
        ss << "let xValue = vec4<f32>(" << dy.GetByOffset("x_offset") << ", " << dy.GetByOffset("x_offset + 1u") << ", " << dy.GetByOffset("x_offset + 2u") << ", " << dy.GetByOffset("x_offset + 3u") << ");\n"
           << "let wValue = vec4<f32>(" << w.GetByOffset("x_offset") << ", " << w.GetByOffset("x_offset + 1u") << ", " << w.GetByOffset("x_offset + 2u") << ", " << w.GetByOffset("x_offset + 3u") << ");\n"
           << "dotProd = dotProd + dot(xValue, wValue);\n"
           << "x_offset += 4;\n"
           << "w_offset += 4;\n";
      }
    } else {
      if (is_channels_last_) {
        ss << "let dy_offset = " << dy.IndicesToOffset("dy_indices_t(batch, idyR, idyC, inputChannel)") << " / " << a_components_ << ";\n"
           << "let xValue = " << dy.GetByOffset("dy_offset") << ";\n";
      } else {
        ss << "let xValue = " << dy.GetByIndices("dy_indices_t(batch, inputChannel, idyR, idyC)") << ";\n";
      }
      if (a_components_ == 1) {
        ss << "let w_offset = " << w.IndicesToOffset("w_indices_t(u32(wRPerm), u32(wCPerm), inputChannel, wOutChannel)") << ";\n"
           << "let wValue = " << w.GetByOffset("w_offset") << ";\n"
           << "dotProd = dotProd + xValue * wValue;\n";
      } else {
        for (uint32_t i = 0; i < a_components_; ++i) {
          ss << "let w_indices = w_indices_t(u32(wRPerm), u32(wCPerm), inputChannel + " << i << ", wOutChannel);\n"
             << "let w_offset = " << w.IndicesToOffset("w_indices") << " / " << b_components_ << ";\n"
             << "let wValue = " << w.GetByOffset("w_offset") << ";\n"
             << "dotProd = dotProd + xValue[" << i << "] * wValue;\n";
        }
      }
    }
    return ss.str();
  };
  auto calculate_remainder = [&]() -> std::string {
    std::stringstream ss;
    if (input_channels_remainder_ > 0) {
      if (a_components_ == 1) {
        for (uint32_t i = 0; i < input_channels_remainder_; ++i) {
          ss << "dotProd = dotProd + " << dy.GetByOffset("x_offset + " + std::to_string(i)) << ";\n";
        }
      } else if (a_components_ == 2) {
        if (input_channels_remainder_ != 2) {
          ORT_THROW("Invalid input_channels_remainder: ", input_channels_remainder_);
        }
        ss << "let xValue = " << dy.GetByOffset("x_offset") << ";\n"
           << "let wValue = " << w.GetByOffset("w_offset") << ";\n"
           << "dotProd = dotProd + dot(xValue, wValue);\n";
      }
    }
    return ss.str();
  };
  shader.MainFunctionBody() << "let outputIndices = " << output.OffsetToIndices("global_idx * " + std::to_string(components_)) << ";\n"
                            << "let batch = " << output.IndicesGet("outputIndices", 0) << ";\n"
                            << "let d1 = " << output.IndicesGet("outputIndices", channel_dim) << ";\n"
                            << "let r = " << output.IndicesGet("outputIndices", row_dim) << ";\n"
                            << "let c = " << output.IndicesGet("outputIndices", col_dim) << ";\n"
                            << "let dyCorner = vec2<i32>(i32(r), i32(c)) - uniforms.pads;\n"
                            << "let dyRCorner = dyCorner.x;\n"
                            << "let dyCCorner = dyCorner.y;\n"
                            << "let groupId = d1 / uniforms.output_channels_per_group;\n"
                            << "let wOutChannel = d1 - groupId * uniforms.output_channels_per_group;\n"
                            << "// Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).\n"
                            << "// ? = to be determined. : = across all values in that axis.\n"
                            << "var dotProd = output_value_t(0.0);\n"
                            << "var wR: u32 = 0;\n"
                            << "if (uniforms.dilations.x == 1) {\n"
                            << "  // Minimum wR >= 0 that satisfies (dyRCorner + wR) % (uniforms.strides.x) == 0\n"
                            << "  wR = u32(((dyRCorner + i32(uniforms.strides.x) - 1) / i32(uniforms.strides.x)) * i32(uniforms.strides.x) - dyRCorner);\n"
                            << "}\n"
                            << "for (; wR < uniforms.effective_filter_dims.x; wR = wR + 1) {\n"
                            << "  if (wR % uniforms.dilations.x != 0) {\n"
                            << "    continue;\n"
                            << "  }\n"
                            << "  let dyR = (x_value_t(dyRCorner) + x_value_t(wR)) / x_value_t(uniforms.strides[0]);\n"
                            << "  let wRPerm = uniforms.filter_dims.x - 1 - wR / uniforms.dilations.x;\n"
                            << "  if (dyR < 0.0 || dyR >= x_value_t(uniforms.Dy_shape[" << row_dim << "]) || fract(dyR) > 0.0 || wRPerm < 0) {\n"
                            << "    continue;\n"
                            << "  }\n"
                            << "  let idyR: u32 = u32(dyR);\n"
                            << "  var wC: u32 = 0;\n"
                            << "  if (uniforms.dilations.y == 1) {\n"
                            << "    // Minimum wC >= 0 that satisfies (dyCCorner + wC) % (uniforms.strides.y) == 0\n"
                            << "    wC = u32(((dyCCorner + i32(uniforms.strides.y) - 1) / i32(uniforms.strides.y)) * i32(uniforms.strides.y) - dyCCorner);\n"
                            << "  }\n"
                            << "  for (; wC < uniforms.effective_filter_dims.y; wC = wC + 1) {\n"
                            << "    if (wC % uniforms.dilations.y != 0) {"
                            << "      continue;\n"
                            << "    }\n"
                            << "    let dyC = (x_value_t(dyCCorner) + x_value_t(wC)) / x_value_t(uniforms.strides.y);\n"
                            << "    let wCPerm = uniforms.filter_dims.y - 1 - wC / uniforms.dilations.y;\n"
                            << "    if (dyC < 0.0 || dyC >= x_value_t(uniforms.Dy_shape[" << col_dim << "]) ||\n"
                            << "        fract(dyC) > 0.0 || wCPerm < 0) {\n"
                            << "      continue;\n"
                            << "    }\n"
                            << "    let idyC: u32 = u32(dyC);\n"
                            << "    var inputChannel = groupId * uniforms.input_channels_per_group;\n";
  if (pack_input_as4_) {
    shader.MainFunctionBody() << "    let dy_indices = dy_indices_t(batch, idyR, idyC, inputChannels);\n"
                              << "    let w_indices = w_indices_t(u32(wRPerm), u32(wCPerm, inputChannel, wOutChannel);\n"
                              << "    var x_offset = " << dy.IndicesToOffset("dy_indices") << " / " << a_components_ << ";\n"
                              << "    var w_offset = " << w.IndicesToOffset("w_indices") << " / " << b_components_ << ";\n";
  }

  shader.MainFunctionBody() << "    for (var d2: u32 = 0; d2 < uniforms.input_channels_per_group_int; d2 = d2 + " << (pack_input_as4_ ? 4 : a_components_) << ") {\n"
                            << "      " << calculate_result() << "\n"
                            << "      inputChannel = inputChannel + " << (pack_input_as4_ ? 4 : a_components_) << ";\n"
                            << "    }\n"
                            << "    " << calculate_remainder() << "\n"
                            << "    wC = wC + uniforms.strides.y - 1;\n"
                            << "  }\n"
                            << "  wR = wR + uniforms.strides.x - 1;\n"
                            << "}\n"
                            << "let value = dotProd" << (has_bias_ ? " + bias[d1 / " + std::to_string(components_) + "]" : "") << "\n"
                            << output.SetByOffset("global_idx", "value") << "\n";
  return Status::OK();
}

ConvTranspose2DProgram CreateConvTranspose2DProgram(const std::vector<const Tensor*>& inputs, const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides, const std::vector<uint32_t>& dilations, Tensor* output, bool is_channels_last, const std::vector<TensorShape>& modified_input_output_shapes, uint32_t groups) {
  bool has_bias = inputs.size() > 2;
  const auto* input = inputs[0];
  const auto* weight = inputs[1];
  const auto& input_shape = modified_input_output_shapes[0];
  const auto& weight_shape = modified_input_output_shapes[1];
  const auto& output_shape = modified_input_output_shapes[has_bias ? 3 : 2];
  auto input_channels_per_group = weight_shape[2] / groups;
  auto output_channels_per_group = weight_shape[3];
  auto a_components = is_channels_last ? GetMaxComponents(input_channels_per_group) : 1;
  bool pack_input_as4 = is_channels_last && output_channels_per_group == 1 && input_channels_per_group >= 4;
  auto input_channels_per_group_int = (input_channels_per_group / 4) * 4;
  auto input_channels_remainder = input_channels_per_group - input_channels_per_group_int;
  auto components = is_channels_last ? GetMaxComponents(output_channels_per_group) : 1;
  auto b_components = is_channels_last ? (output_channels_per_group == 1 ? a_components : components) : 1;
  auto input_shape_vector = input_shape.AsShapeVector();
  input_shape_vector[input_shape_vector.size() - 1] /= a_components;
  TensorShape reduced_input_shape(input_shape_vector);
  InlinedVector<int64_t> weight_shape_vector = weight_shape.AsShapeVector();
  weight_shape_vector[weight_shape_vector.size() - 1] /= components;
  TensorShape reduced_weight_shape(weight_shape_vector);
  InlinedVector<int64_t> output_shape_vector = output_shape.AsShapeVector();
  output_shape_vector[output_shape_vector.size() - 1] /= components;
  TensorShape reduced_output_shape(output_shape_vector);
  auto output_size = output_shape.Size() / components;
  std::vector<uint32_t> filter_dims = {static_cast<uint32_t>(weight_shape[is_channels_last ? 1 : 2]), static_cast<uint32_t>(weight_shape[is_channels_last ? 2 : 3])};
  std::vector<uint32_t> effective_filter_dims = {filter_dims[0] + dilations[0] <= 1 ? 0 : filter_dims[0] * (dilations[0] - 1), filter_dims[1] + dilations[1] <= 1 ? 0 : filter_dims[1] * (dilations[1] - 1)};
  std::vector<uint32_t> local_pads = {effective_filter_dims[0] - 1 - (pads[0] + pads[1]) / 2, effective_filter_dims[1] - 1 - (pads[1] + pads[3]) / 2};
  ConvTranspose2DProgram program(is_channels_last, has_bias, components, a_components, b_components, uint32_t(input_channels_remainder), pack_input_as4);
  program.AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, a_components}, {weight, ProgramTensorMetadataDependency::TypeAndRank, reduced_weight_shape, components}});
  if (has_bias) {
    const auto* bias = inputs[2];
    const auto& bias_shape = modified_input_output_shapes[2];
    TensorShapeVector bias_shape_vector = bias_shape.AsShapeVector();
    bias_shape_vector[bias_shape_vector.size() - 1] /= components;
    TensorShape reduced_bias_shape(bias_shape_vector);
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, reduced_bias_shape, components});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::Rank, reduced_output_shape, components});
  program.AddUniformVariables({{static_cast<uint32_t>(output_size)}, {strides}, {filter_dims}, {dilations}, {effective_filter_dims}, {pads}, {static_cast<uint32_t>(input_channels_per_group)}, {static_cast<uint32_t>(input_channels_per_group_int)}, {static_cast<uint32_t>(output_channels_per_group)}});
  return program;
}

}  // namespace webgpu
}  // namespace onnxruntime
