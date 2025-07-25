// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include "core/providers/webgpu/nn/conv2d_mm_webgpu.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/activation_util.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/nn/conv_utils.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"

namespace onnxruntime {
namespace webgpu {
std::string Conv2dMMProgram::Conv2dCommonSnippet(const ShaderVariableHelper& x, const ShaderVariableHelper& w, const Activation& activation, std::string data_type, uint32_t inner_element_size_x, uint32_t inner_element_size_w, uint32_t inner_element_size) const {
  auto get_x_snippet = [&](int32_t inner_element_size) -> std::string {
    switch (inner_element_size) {
      case 1:
        return "resData = " + x.GetByOffset("xIndex") + ";";
      case 3:
        return "resData = vec3<x_element_t>(" + x.GetByOffset("xIndex") + ", " + x.GetByOffset("xIndex + 1") + ", " + x.GetByOffset("xIndex + 2") + ");";
      case 4:
        return "resData = " + x.GetByOffset("xIndex") + ";\n ";
      default:
        ORT_THROW("inner_element_size", inner_element_size, " is not supported.");
    }
  };
  auto get_w_snippet = [&](int32_t inner_element_size) -> std::string {
    switch (inner_element_size) {
      case 1:
        return "return " + w.GetByOffset("row * i32(uniforms.w_shape[3]) + colIn") + ";\n";
      case 4:
        return "return " + w.GetByOffset("row * i32(uniforms.w_shape[3])  + colIn") + ";\n";
      default:
        ORT_THROW("inner_element_size ", inner_element_size, " is not supported.");
    }
  };
  const std::string coord_a_snippet = is_channels_last_ ? "let coord = vec4<i32>(batch, xRow, xCol, xCh / " + std::to_string(inner_element_size_x == 3 ? 4 : inner_element_size_x) + ");" : "let coord = vec4<i32>(batch, xCh, xRow, xCol);";
  const std::string coord_res_snippet = is_channels_last_ ? "let coords = vec4<i32>(batch, row / outWidth, row % outWidth, col / " + std::to_string(inner_element_size) + ");" : "let coords = vec4<i32>(batch, row, col / outWidth, col % outWidth);";

  const std::string xHeight = is_channels_last_ ? "i32(uniforms.x_shape[1])" : "i32(uniforms.x_shape[2])";
  const std::string xWidth = is_channels_last_ ? "i32(uniforms.x_shape[2])" : "i32(uniforms.x_shape[3])";
  const std::string row = is_channels_last_ ? "row" : "col";
  const std::string col = is_channels_last_ ? "col" : "row";
  std::stringstream read_x_snippet;
  read_x_snippet
      << "let inChannels = i32(uniforms.w_shape[2]);\n"
      << "let outWidth = " << (is_channels_last_ ? "i32(uniforms.result_shape[2])" : "i32(uniforms.result_shape[3])") << ";\n"
      << "let outRow = " << row << " / outWidth;\n "
      << "let outCol = " << row << " % outWidth;\n"
      << "let WRow = " << col << " / (i32(uniforms.w_shape[1]) * inChannels);\n"
      << "let WCol = " << col << " / inChannels % i32(uniforms.w_shape[1]);\n"
      << "let xRow = outRow * i32(uniforms.strides[0]) + i32(uniforms.dilations[0]) * WRow - i32(uniforms.pads[0]);\n"
      << "let xCol = outCol * i32(uniforms.strides[1]) + i32(uniforms.dilations[1]) * WCol - i32(uniforms.pads[1]);\n"
      << "let xCh = " << col << " % inChannels;\n"
      << "var resData = " << TypeSnippet(inner_element_size_x, data_type) << "(0.0);\n "
      << "// The bounds checking is always needed since we use it to pad zero for\n"
      << "// the \" same \" padding type.\n"
      << "if (xRow >= 0 && xRow < " << xHeight << " && xCol >= 0 && xCol < " << xWidth << ") {\n"
      << "  " << coord_a_snippet << "\n"
      << "  let xIndex = getIndexFromCoords4D(coord, vec4<i32>(uniforms.x_shape));\n"
      << "  " << get_x_snippet(inner_element_size_x)
      << "}\n"
      << "return resData;";
  std::stringstream sample_x;
  if (is_channels_last_) {
    if (fit_a_outer_ && fit_inner_) {
      sample_x << "let col = colIn * " << inner_element_size_x << ";\n"
               << read_x_snippet.str();
    } else {
      sample_x << "let col = colIn * " << inner_element_size_x << ";\n"
               << "if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_inner)) {\n"
               << "  " << read_x_snippet.str() << "\n"
               << "}\n"
               << "return " << TypeSnippet(inner_element_size_x, data_type) << "(0.0);\n";
    }
  } else {
    if (fit_inner_ && fit_b_outer_) {
      sample_x << "let col = colIn * " << inner_element_size_x << ";\n"
               << read_x_snippet.str();
    } else {
      sample_x << "let col = colIn * " << inner_element_size_x << ";\n"
               << "if (row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {\n"
               << "  " << read_x_snippet.str() << "\n"
               << "}\n"
               << "return " << TypeSnippet(inner_element_size_x, data_type) << "(0.0);\n";
    }
  }
  std::stringstream sample_w;
  if (is_channels_last_) {
    if (fit_inner_ && fit_b_outer_) {
      sample_w << get_w_snippet(inner_element_size_w);
    } else {
      sample_w << "let col = colIn * " << inner_element_size_w << ";\n"
               << "if(row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {\n"
               << "  " << get_w_snippet(inner_element_size_w) << "\n"
               << "}\n"
               << "return " << TypeSnippet(inner_element_size_w, data_type) << "(0.0);\n";
    }
  } else {
    sample_w << "let col = colIn * " << inner_element_size_w << ";\n"
             << "if (row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {\n"
             << "  " << get_w_snippet(inner_element_size_w) << "\n"
             << "}\n"
             << "return " << TypeSnippet(inner_element_size_w, data_type) << "(0.0);\n";
  }
  const std::string res_type = TypeSnippet(inner_element_size, data_type);
  const std::string a_type = is_channels_last_ ? TypeSnippet(inner_element_size_x, data_type) : TypeSnippet(inner_element_size_w, data_type);
  const std::string b_type = is_channels_last_ ? TypeSnippet(inner_element_size_w, data_type) : TypeSnippet(inner_element_size_x, data_type);
  const std::string apply_activation = GetActivationSnippet(activation, res_type, data_type);
  std::stringstream user_code;
  user_code << "fn mm_readA(batch : i32, row : i32, colIn : i32) -> " << a_type << " {\n"
            << (is_channels_last_ ? sample_x.str() : sample_w.str())
            << "}\n"
            << "\n"
            << "fn mm_readB(batch : i32, row : i32, colIn : i32) -> " << b_type << " {\n"
            << (is_channels_last_ ? sample_w.str() : sample_x.str())
            << "}\n"
            << "\n"
            << "fn mm_write(batch : i32, row : i32, colIn : i32, valueIn : " << res_type << ") {\n"
            << "  let col = colIn * " << inner_element_size << ";\n"
            << "  if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) {\n"
            << "    var value = valueIn;\n"
            << "    let outWidth = " << (is_channels_last_ ? " i32(uniforms.result_shape[2]) " : " i32(uniforms.result_shape[3]) ") << ";\n"
            << "    " << coord_res_snippet << "\n"
            << "    " << BiasSnippet(has_bias_) << "\n"
            << "    " << apply_activation << "\n"
            << "    setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);\n"
            << "  }\n"
            << "}\n";
  return user_code.str();
}

Status Conv2dMMProgram::GenerateShaderCode(ShaderHelper& shader) const {
  std::stringstream declaration_functions;
  declaration_functions << "fn setOutputAtIndex(flatIndex : i32, value : " << (is_vec4_ ? "vec4<x_element_t>" : "x_element_t") << ") {\n"
                        << "  result[flatIndex] = " << (is_vec4_ ? "vec4<x_element_t>" : "x_element_t") << "(value);\n"
                        << "}\n"
                        << "fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : " << (is_vec4_ ? "vec4<x_element_t>" : "x_element_t") << "){\n"
                        << "  let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));\n"
                        << "  setOutputAtIndex(flatIndex, value);\n"
                        << "}\n";
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  std::vector<const ShaderVariableHelper*> inputs = {&x, &w};
  ORT_IGNORE_RETURN_VALUE(shader.AddOutput("result", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseIndicesTypeAlias));
  if (has_bias_) {
    const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    inputs.push_back(&bias);
    declaration_functions << "fn getBiasByOutputCoords(coords : vec4<i32>) -> bias_value_t {" << "\n"
                          << "  return bias[" << (is_channels_last_ ? "coords.w" : "coords.y") << "];\n"
                          << "}";
  }
  shader.AdditionalImplementation()
      << UtilFunctions("uniforms.result_stride")
      << declaration_functions.str()
      << Conv2dCommonSnippet(x, w, activation_, "x_element_t", element_size_[0], element_size_[1], element_size_[2]);
  std::string data_type = "x_element_t";

  return is_vec4_ ? MakeMatMulPackedVec4Source(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, /* batch_dims = */ nullptr, /* transpose_a = */ !is_channels_last_, /* transpose_b = */ false, 1.0f, true, 4, tile_inner_)
                  : MakeMatMulPackedSource(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, /* batch_dims = */ nullptr, /*transpose_a = */ !is_channels_last_, /* transpose_b = */ false, 1.0f, true, tile_inner_, /* split_t = */ false, 0);
}

Conv2dMMProgram CreateConv2dMMProgram(const Activation& activation, const std::vector<const Tensor*>& inputs, const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides, const std::vector<uint32_t>& dilations, Tensor* output, uint32_t dim_a_outer, uint32_t dim_b_outer, uint32_t dim_inner, bool is_channels_last, const std::vector<TensorShape>& input_output_shapes) {
  const auto* input = inputs[0];
  const auto* weight = inputs[1];
  bool has_bias = inputs.size() > 2;
  const auto* bias = has_bias ? inputs[2] : nullptr;
  const auto& input_shape = input_output_shapes[0];
  auto in_channels = is_channels_last ? input_shape[3] : input_shape[1];
  const auto& output_shape = has_bias ? input_output_shapes[3] : input_output_shapes[2];
  auto batch_size = output_shape[0];
  const auto output_width = is_channels_last ? output_shape[2] : output_shape[3];
  const auto output_height = is_channels_last ? output_shape[1] : output_shape[2];
  const auto output_channels = is_channels_last ? output_shape[3] : output_shape[1];
  // TODO: enable vec4 for NCHW
  const bool is_vec4 = is_channels_last && (in_channels % 4 == 0 || in_channels % 3 == 0) && output_channels % 4 == 0;

  // TODO: fine tune size
  const auto dispatch_x = is_channels_last ? output_channels : output_width * output_height;
  const auto dispatch_y = is_channels_last ? output_width * output_height : output_channels;
  std::vector<uint32_t> workgroup_size = {8, 8, 1};
  InlinedVector<int64_t> elements_per_thread = {4, static_cast<int64_t>(dim_a_outer <= 8 ? 1 : 4), 1};
  auto integer_ceil = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

  const std::vector<uint32_t> dispatch = {
      static_cast<uint32_t>(integer_ceil(integer_ceil(dispatch_x, workgroup_size[0]), elements_per_thread[0])),
      static_cast<uint32_t>(integer_ceil(integer_ceil(dispatch_y, workgroup_size[1]), elements_per_thread[1])),
      static_cast<uint32_t>(integer_ceil(integer_ceil(batch_size, workgroup_size[2]), elements_per_thread[2])),
  };

  uint32_t inner_element_size = is_vec4 ? (is_channels_last && in_channels % 4 != 0 ? 3 : 4) : 1;
  auto tile_a_outer = static_cast<uint32_t>(workgroup_size[1] * elements_per_thread[1]);
  auto tile_b_outer = static_cast<uint32_t>(workgroup_size[0] * elements_per_thread[0]);
  auto tile_inner = std::max(workgroup_size[0] * inner_element_size, workgroup_size[1]);
  bool fit_a_outer = dim_a_outer % tile_a_outer == 0;
  bool fit_b_outer = dim_b_outer % tile_b_outer == 0;
  bool fit_inner = dim_inner % tile_inner == 0;
  std::vector<uint32_t> element_size = {is_vec4 ? inner_element_size : 1, static_cast<uint32_t>(is_vec4 ? 4 : 1), static_cast<uint32_t>(is_vec4 ? 4 : 1)};
  const auto components = is_vec4 ? 4 : 1;
  const auto input_components = static_cast<int>(inner_element_size == 3 ? 1 : inner_element_size);
  Conv2dMMProgram program(activation, tile_inner, fit_a_outer, fit_b_outer, fit_inner, is_channels_last, is_vec4, has_bias, std::move(element_size), std::move(elements_per_thread));
  TensorShape reduced_input_shape = ReduceShapeByComponents(input_output_shapes[0], input_components);
  TensorShape reduced_weight_shape = ReduceShapeByComponents(input_output_shapes[1], components);
  TensorShape reduced_output_shape = ReduceShapeByComponents(input_output_shapes[has_bias ? 3 : 2], components);
  program.AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, input_components}, {weight, ProgramTensorMetadataDependency::TypeAndRank, reduced_weight_shape, components}});
  if (has_bias) {
    TensorShape reduced_bias_shape = ReduceShapeByComponents(input_output_shapes[2], components);
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, reduced_bias_shape, components});
  }
  const auto stringify = [](const std::vector<uint32_t>& vec) -> std::string {
    std::ostringstream oss;
    std::transform(vec.begin(), vec.end(), std::ostream_iterator<std::string>(oss, ","), [](uint32_t i) { return std::to_string(i); });
    return oss.str();
  };
  program.CacheHint(activation.ToString(), is_channels_last, stringify({inner_element_size, static_cast<uint32_t>(is_vec4 ? 1 : 0), fit_a_outer, fit_b_outer, fit_inner, tile_a_outer, tile_a_outer, tile_inner, static_cast<uint32_t>(components)}))
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, components})
      .SetDispatchGroupSize(dispatch[0], dispatch[1], dispatch[2])
      .SetWorkgroupSize(workgroup_size[0], workgroup_size[1], workgroup_size[2])
      .AddUniformVariables({{static_cast<uint32_t>(dim_a_outer)},
                            {static_cast<uint32_t>(dim_b_outer)},
                            {static_cast<uint32_t>(dim_inner)},
                            {pads},
                            {strides},
                            {dilations}});

  return program;
}

}  // namespace webgpu
}  // namespace onnxruntime
