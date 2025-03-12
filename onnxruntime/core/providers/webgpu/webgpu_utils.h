// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace webgpu {

inline int64_t GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

}  // namespace webgpu
}  // namespace onnxruntime