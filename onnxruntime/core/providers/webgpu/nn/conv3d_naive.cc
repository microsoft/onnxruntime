// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/providers/webgpu/nn/conv3d_naive.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime {
namespace webgpu {

Status Conv3DNaiveProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias |
                                           ShaderUsage::UseElementTypeAlias);
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias |
                                                      ShaderUsage::UseValueTypeAlias |
                                                      ShaderUsage::UseElementTypeAlias);

  std::string apply_activation = GetActivationSnippet(activation_, "x_value_t", "x_element_t");

  // Helper functions to access x and w by 5D indices
  shader.AdditionalImplementation()
      << "fn getX(d0 : u32, d1 : u32, d2 : u32, d3 : u32, d4 : u32) -> x_value_t {\n"
      << "  let aIndices = x_indices_t(d0, d1, d2, d3, d4);\n"
      << "  return " << x.GetByIndices("aIndices") << ";\n"
      << "}\n"
      << "fn getW(d0 : u32, d1 : u32, d2 : u32, d3 : u32, d4 : u32) -> x_value_t {\n"
      << "  let aIndices = w_indices_t(d0, d1, d2, d3, d4);\n"
      << "  return " << w.GetByIndices("aIndices") << ";\n"
      << "}\n";

  // Spatial dimensions and channels are passed as explicit uniforms
  // to avoid rank-5 shape packing issues (array<vec4<u32>,2> vs vec4<u32>).
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
      << "let batch = output_indices[0];\n"
      << "let d2 = " << output.IndicesGet("output_indices", is_channels_last_ ? "4" : "1") << ";\n"
      << "let xFRCCorner = vec3<u32>(" << output.IndicesGet("output_indices", is_channels_last_ ? "1" : "2") << ", "
      << output.IndicesGet("output_indices", is_channels_last_ ? "2" : "3") << ", "
      << output.IndicesGet("output_indices", is_channels_last_ ? "3" : "4") << ") * uniforms.strides - uniforms.pads;\n"
      << "let xFCorner = xFRCCorner.x;\n"
      << "let xRCorner = xFRCCorner.y;\n"
      << "let xCCorner = xFRCCorner.z;\n"
      << "let xDepth = uniforms.x_spatial[0];\n"
      << "let xHeight = uniforms.x_spatial[1];\n"
      << "let xWidth = uniforms.x_spatial[2];\n"
      << "let xChannels = uniforms.x_channels;\n"
      << "let inputChannelsNearestVec4 = (xChannels / 4u) * 4u;\n"
      << "let inputChannelsVec4Remainder = xChannels % 4u;\n"
      << "\n"
      << "var value = x_value_t(0);\n"
      << "for (var wF = 0u; wF < uniforms.filter_dims[0]; wF++) {\n"
      << "  let xF = xFCorner + wF * uniforms.dilations[0];\n"
      << "  if (xF >= xDepth) {\n"
      << "    continue;\n"
      << "  }\n"
      << "  for (var wR = 0u; wR < uniforms.filter_dims[1]; wR++) {\n"
      << "    let xR = xRCorner + wR * uniforms.dilations[1];\n"
      << "    if (xR >= xHeight) {\n"
      << "      continue;\n"
      << "    }\n"
      << "    for (var wC = 0u; wC < uniforms.filter_dims[2]; wC++) {\n"
      << "      let xC = xCCorner + wC * uniforms.dilations[2];\n"
      << "      if (xC >= xWidth) {\n"
      << "        continue;\n"
      << "      }\n"
      << "      for (var d1 = 0u; d1 < inputChannelsNearestVec4; d1 += 4u) {\n";

  // vec4 dot product accumulation over input channels
  if (is_channels_last_) {
    shader.MainFunctionBody()
        << "        let xValues = vec4<x_element_t>(\n"
        << "            getX(batch, xF, xR, xC, d1),\n"
        << "            getX(batch, xF, xR, xC, d1 + 1u),\n"
        << "            getX(batch, xF, xR, xC, d1 + 2u),\n"
        << "            getX(batch, xF, xR, xC, d1 + 3u));\n";
  } else {
    shader.MainFunctionBody()
        << "        let xValues = vec4<x_element_t>(\n"
        << "            getX(batch, d1, xF, xR, xC),\n"
        << "            getX(batch, d1 + 1u, xF, xR, xC),\n"
        << "            getX(batch, d1 + 2u, xF, xR, xC),\n"
        << "            getX(batch, d1 + 3u, xF, xR, xC));\n";
  }
  shader.MainFunctionBody()
      << "        let wValues = vec4<x_element_t>(\n"
      << "            getW(d2, d1, wF, wR, wC),\n"
      << "            getW(d2, d1 + 1u, wF, wR, wC),\n"
      << "            getW(d2, d1 + 2u, wF, wR, wC),\n"
      << "            getW(d2, d1 + 3u, wF, wR, wC));\n"
      << "        value += x_value_t(dot(xValues, wValues));\n"
      << "      }\n";

  // Handle remainder channels (1, 2, or 3)
  shader.MainFunctionBody()
      << "      if (inputChannelsVec4Remainder == 1u) {\n";
  if (is_channels_last_) {
    shader.MainFunctionBody()
        << "        value += getX(batch, xF, xR, xC, inputChannelsNearestVec4)\n"
        << "            * getW(d2, inputChannelsNearestVec4, wF, wR, wC);\n";
  } else {
    shader.MainFunctionBody()
        << "        value += getX(batch, inputChannelsNearestVec4, xF, xR, xC)\n"
        << "            * getW(d2, inputChannelsNearestVec4, wF, wR, wC);\n";
  }
  shader.MainFunctionBody()
      << "      } else if (inputChannelsVec4Remainder == 2u) {\n";
  if (is_channels_last_) {
    shader.MainFunctionBody()
        << "        let xValues = vec2<x_element_t>(\n"
        << "            getX(batch, xF, xR, xC, inputChannelsNearestVec4),\n"
        << "            getX(batch, xF, xR, xC, inputChannelsNearestVec4 + 1u));\n";
  } else {
    shader.MainFunctionBody()
        << "        let xValues = vec2<x_element_t>(\n"
        << "            getX(batch, inputChannelsNearestVec4, xF, xR, xC),\n"
        << "            getX(batch, inputChannelsNearestVec4 + 1u, xF, xR, xC));\n";
  }
  shader.MainFunctionBody()
      << "        let wValues = vec2<x_element_t>(\n"
      << "            getW(d2, inputChannelsNearestVec4, wF, wR, wC),\n"
      << "            getW(d2, inputChannelsNearestVec4 + 1u, wF, wR, wC));\n"
      << "        value += x_value_t(dot(xValues, wValues));\n"
      << "      } else if (inputChannelsVec4Remainder == 3u) {\n";
  if (is_channels_last_) {
    shader.MainFunctionBody()
        << "        let xValues = vec3<x_element_t>(\n"
        << "            getX(batch, xF, xR, xC, inputChannelsNearestVec4),\n"
        << "            getX(batch, xF, xR, xC, inputChannelsNearestVec4 + 1u),\n"
        << "            getX(batch, xF, xR, xC, inputChannelsNearestVec4 + 2u));\n";
  } else {
    shader.MainFunctionBody()
        << "        let xValues = vec3<x_element_t>(\n"
        << "            getX(batch, inputChannelsNearestVec4, xF, xR, xC),\n"
        << "            getX(batch, inputChannelsNearestVec4 + 1u, xF, xR, xC),\n"
        << "            getX(batch, inputChannelsNearestVec4 + 2u, xF, xR, xC));\n";
  }
  shader.MainFunctionBody()
      << "        let wValues = vec3<x_element_t>(\n"
      << "            getW(d2, inputChannelsNearestVec4, wF, wR, wC),\n"
      << "            getW(d2, inputChannelsNearestVec4 + 1u, wF, wR, wC),\n"
      << "            getW(d2, inputChannelsNearestVec4 + 2u, wF, wR, wC));\n"
      << "        value += x_value_t(dot(xValues, wValues));\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "}\n";

  // Apply bias
  if (has_bias_) {
    const auto& b = shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody() << "value = value + " << b.GetByIndices("d2") << ";\n";
  }

  // Apply activation
  shader.MainFunctionBody() << apply_activation << "\n";

  // Write output
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
