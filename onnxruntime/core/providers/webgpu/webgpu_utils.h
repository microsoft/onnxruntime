// Licensed under the MIT License.

#pragma once

#include <cstdint>

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

}  // namespace webgpu
}  // namespace onnxruntime
