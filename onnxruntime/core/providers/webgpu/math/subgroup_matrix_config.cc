// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/math/subgroup_matrix_config.h"

#include <cstddef>

namespace onnxruntime {
namespace webgpu {

bool IsSubgroupMatrixConfigSupported(const ComputeContextBase& context, bool is_fp16, int32_t& config_index) {
  if (!context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    return false;
  }
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();
  int32_t index = 0;
  for (const auto& supported_config : supported_subgroup_matrix_configs) {
    // F16 configs require FP16 output; skip them when output is F32.
    // F32 configs require FP32 output; skip them when output is FP16.
    if ((supported_config.componentType == wgpu::SubgroupMatrixComponentType::F16 && !is_fp16) ||
        (supported_config.componentType == wgpu::SubgroupMatrixComponentType::F32 && is_fp16)) {
      index++;
      continue;
    }
    for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
      const auto& device_config = subgroup_matrix_configs.configs[i];
      if (device_config.componentType == supported_config.componentType &&
          device_config.resultComponentType == supported_config.resultComponentType &&
          device_config.M == supported_config.M &&
          device_config.N == supported_config.N &&
          device_config.K == supported_config.K &&
          adapter_info.subgroupMinSize == supported_config.subgroupMinSize &&
          adapter_info.subgroupMaxSize == supported_config.subgroupMaxSize) {
        config_index = index;
        return true;
      }
    }
    index++;
  }
  return false;
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
