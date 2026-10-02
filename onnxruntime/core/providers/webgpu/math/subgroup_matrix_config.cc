// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_config.h"

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {
namespace detail {

std::optional<SubgroupMatrixConfig> SelectSubgroupMatrixConfigFromAdapterConfigs(
    gsl::span<const wgpu::SubgroupMatrixConfig> adapter_configs,
    uint32_t adapter_min_subgroup_size,
    uint32_t adapter_max_subgroup_size,
    bool has_subgroup_size_control,
    std::initializer_list<SubgroupMatrixConfig> preferences) {
  for (const auto& preference : preferences) {
    if (!IsSubgroupSizeSupported(adapter_min_subgroup_size, adapter_max_subgroup_size,
                                 preference.subgroupSize, has_subgroup_size_control)) {
      continue;
    }
    for (const auto& adapter_config : adapter_configs) {
      if (adapter_config.componentType == preference.componentType &&
          adapter_config.resultComponentType == preference.resultComponentType &&
          adapter_config.M == preference.M &&
          adapter_config.N == preference.N &&
          adapter_config.K == preference.K) {
        return preference;
      }
    }
  }
  return std::nullopt;
}

}  // namespace detail

std::optional<SubgroupMatrixConfig> SelectSubgroupMatrixConfig(
    const ComputeContextBase& context,
    std::initializer_list<SubgroupMatrixConfig> preferences) {
  if (!context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    return std::nullopt;
  }

  const auto& adapter_info = context.AdapterInfo();
  const auto& adapter_configs = context.SubgroupMatrixConfigs();
  return detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
      {adapter_configs.configs, adapter_configs.configCount},
      adapter_info.subgroupMinSize,
      adapter_info.subgroupMaxSize,
      context.HasFeature(wgpu::FeatureName::SubgroupSizeControl),
      preferences);
}

}  // namespace webgpu
}  // namespace onnxruntime
