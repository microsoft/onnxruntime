// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/conv_utils.h"
namespace onnxruntime {
namespace webgpu {
std::string UtilFunctions(std::string stride_string) {
  std::stringstream ss;
  ss << "fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {\n"
     << "  return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));\n"
     << "}\n"
     << "fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {\n"
     << "  return dot(coords, vec4<i32>(shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));\n"
     << "}\n"
     << "fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {\n"
     << "  return dot(coords, vec4<i32>(i32(" << stride_string << ".x), i32(" << stride_string << ".y), i32(" << stride_string << ".z), 1));\n"
     << "}\n";
  return ss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
