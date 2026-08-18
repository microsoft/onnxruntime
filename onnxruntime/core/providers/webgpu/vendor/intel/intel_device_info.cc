// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/vendor/intel/intel_device_info.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

uint32_t HwSubgroups(std::string_view arch) {
  if (arch == std::string_view{"xe-3lpg"}) {
    return 384;  // 12 Xe cores x 32
  }
  if (arch == std::string_view{"xe-2lpg"}) {
    return 256;  // 8 Xe cores x 32
  }
  return 0;  // unknown architecture; caller decides the fallback
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
