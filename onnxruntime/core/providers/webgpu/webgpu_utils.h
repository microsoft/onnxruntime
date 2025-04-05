// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "core/common/common.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace webgpu {

inline int GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

inline std::string SumVector(std::string x, int components) {
  switch (components) {
    case 1:
      return x;
    case 2:
      return "(" + x + ".x + " + x + ".y" + ")";
    case 4:
      return "(" + x + ".x + " + x + ".y + " + x + ".z + " + x + ".w" + ")";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

inline std::string MakeScalarOrVectorType(int components, std::string_view data_type) {
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

TensorShape ReduceShapeByComponents(const TensorShape& shape, int64_t components);

}  // namespace webgpu
}  // namespace onnxruntime
