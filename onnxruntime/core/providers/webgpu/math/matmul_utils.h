// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

static int GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

inline TensorShape BuildTempShapeVector(const TensorShape& shape, const int64_t dim1, const int64_t dim2, const int components) {
    TensorShapeVector shape_vec = shape.AsShapeVector();
    shape_vec.push_back(dim1);
    shape_vec.push_back(dim2 / components);
    return TensorShape(shape_vec);
}


// write our own methods for i2o and o2i, since we don't optimize for stride
// const std::string MatMulI2o(const ShaderVariableHelper& input, const std::string& indices) {
//     return MakeStringWithClassicLocale("i2o_", input.Name(), '(', indices, ')');
// }

// Helper that convert output batch indices to input batch indices using only the rank and
// the shape information in uniform
inline std::string ConvertOutputBatchIndicesToInputBatchIndices(const std::string& name, const ShaderVariableHelper& input, int input_batch_rank, int output_batch_rank, const std::string& batch_indices) {
    std::ostringstream oss;
    const std::string input_shape = "uniforms." + name + "_shape";
    const std::string input_indices = name + "_indices";
    int extending_input_rank = output_batch_rank - input_batch_rank;
    for (int i = 0; i < input_batch_rank; ++i) {
        oss << "if (" << GetElementAt(input_shape, i, input.Rank()) << " != 1) {\n"
            << input.IndicesSet(input_indices, i, GetElementAt(batch_indices, i + extending_input_rank, output_batch_rank)) << "\n"
            << "} else {\n"
            << input.IndicesSet(input_indices, i, 0) << "\n"
            << "}\n";
    }
    return oss.str();
}

inline std::string MakeTypeString(int components, std::string_view data_type) {
  switch (components) {
    case 1:
      return std::string{data_type};
    case 2:
      return MakeStringWithClassicLocale("vec2<", data_type, ">");
    case 3:
      return MakeStringWithClassicLocale("vec3<", data_type, ">");
    case 4:
      return MakeStringWithClassicLocale("vec4<", data_type, ">");
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

}  // namespace webgpu
} // namespace onnxruntime
