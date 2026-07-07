// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/vendor/intel/intel_device_info.h"

#include <algorithm>

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

bool IsSubgroupMatrixConfigSupported(const ComputeContextBase& context, uint32_t m, uint32_t n, uint32_t k) {
  if (!context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    return false;
  }
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  if (adapter_info.subgroupMinSize != 16 || adapter_info.subgroupMaxSize != 32) {
    return false;
  }
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& configs = context.SubgroupMatrixConfigs();
  return std::any_of(configs.configs, configs.configs + configs.configCount,
                     [m, n, k](const auto& c) {
                       return c.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                              c.resultComponentType == wgpu::SubgroupMatrixComponentType::F16 &&
                              c.M == m && c.N == n && c.K == k;
                     });
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
