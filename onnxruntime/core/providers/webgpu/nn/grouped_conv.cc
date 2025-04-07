// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <sstream>
#include "core/providers/webgpu/nn/grouped_conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

std::string CanculateResult(const ShaderVariableHelper& x, const ShaderVariableHelper& w, bool is_channels_last) {
  std::stringstream ss;
  if (is_channels_last) {
    ss << "for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[0]; wHeight++) {\n"
       << "    let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];\n"
       << "    if (xHeight < 0u || xHeight >= uniforms.x_shape[1]) {\n"
       << "      continue;\n"
       << "    }\n"
       << ""
       << "    for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[1]; wWidth++) {\n"
       << "      let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];\n"
       << "      if (xWidth < 0u || xWidth >= uniforms.x_shape[2]) {\n"
       << "        continue;\n"
       << "      }\n"
       << ""
       << "    for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[2]; wInChannel++) {\n"
       << "      let input_channel = in_channel_offset + wInChannel;\n"
       << "      let x_indices = x_indices_t(batch, xHeight, xWidth, input_channel);\n"
       << "      let w_indices = w_indices_t(wHeight, wWidth, wInChannel, output_channel);\n"
       << "      let xVal = " << x.GetByIndices("x_indices") << ";\n"
       << "      let wVal = " << w.GetByIndices("w_indices") << ";\n"
       << "      value += xVal * wVal;\n"
       << "    }\n"
       << "  }\n"
       << "}\n";
  } else {
    ss << "for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[1]; wInChannel++) {\n"
       << "  let input_channel = in_channel_offset + wInChannel;\n"
       << "  for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[2]; wHeight++) {\n"
       << "    let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];\n"
       << ""
       << "    if (xHeight < 0u || xHeight >= uniforms.x_shape[2]) {\n"
       << "      continue;\n"
       << "    }\n"
       << ""
       << "    for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[3]; wWidth++) {\n"
       << "      let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];\n"
       << "      if (xWidth < 0u || xWidth >= uniforms.x_shape[3]) {\n"
       << "        continue;\n"
       << "      }\n"
       << ""
       << "      let x_indices = x_indices_t(batch, input_channel, xHeight, xWidth);\n"
       << "      let w_indices = w_indices_t(output_channel, wInChannel, wHeight, wWidth);\n"
       << "      let xVal = " << x.GetByIndices("x_indices") << ";\n"
       << "      let wVal = " << w.GetByIndices("w_indices") << ";\n"
       << "      value += xVal * wVal;\n"
       << "    }\n"
       << "  }\n"
       << "}\n";
  }
  return ss.str();
}

Status GroupedConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "let batch: u32 = output_indices[0];\n"
                            << "let output_channel: u32 = " << output.IndicesGet("output_indices", is_channels_last_ ? "3" : "1") << ";\n"
                            << "let xRCCorner_x: u32 = " << output.IndicesGet("output_indices", is_channels_last_ ? "1" : "2") << ";\n"
                            << "let xRCCorner_y: u32 = " << output.IndicesGet("output_indices", is_channels_last_ ? "2" : "3") << ";\n"
                            << "let xRCCorner: vec2<u32> = vec2<u32>(xRCCorner_x, xRCCorner_y) * uniforms.strides - uniforms.pads;\n"
                            << "let group_id = output_channel * uniforms.components / uniforms.output_channels_per_group;\n"
                            << "let in_channel_offset = group_id * " << w.IndicesGet("uniforms.w_shape", is_channels_last_ ? 2 : 1) << ";\n"
                            << "var value: output_value_t = output_value_t(0);\n"
                            << CanculateResult(x, w, is_channels_last_);
  if (has_bias_) {
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody() << "value += " + b.GetByIndices("output_channel") + ";\n";
  }
  shader.MainFunctionBody() << apply_activation << "\n";
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "value");
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
