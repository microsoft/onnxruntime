// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_config.h"

#include <cstddef>

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {
namespace {

using SubgroupMatrixConfigIndices = InlinedVector<int32_t, 8>;

SubgroupMatrixConfigIndices GetSupportedSubgroupMatrixConfigIndices(
    const ComputeContextBase& context,
    bool is_fp16) {
  SubgroupMatrixConfigIndices candidates;
  if (!context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    return {};
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
          IsSubgroupSizeSupported(adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize,
                                  supported_config.subgroupSize,
                                  context.HasFeature(wgpu::FeatureName::SubgroupSizeControl))) {
        candidates.push_back(index);
        break;
      }
    }
    index++;
  }
  return candidates;
}

}  // namespace

namespace detail {

std::optional<int32_t> SelectSubgroupMatrixConfigFromCandidates(
    gsl::span<const int32_t> candidate_indices,
    std::initializer_list<SubgroupMatrixConfigPreference> preferences) {
  for (const auto& preference : preferences) {
    for (const int32_t index : candidate_indices) {
      if (index < 0 || static_cast<size_t>(index) >= supported_subgroup_matrix_configs.size()) {
        continue;
      }
      const auto& config = supported_subgroup_matrix_configs[index];
      if (config.Is(preference.M, preference.N, preference.K) &&
          config.subgroupSize == preference.subgroupSize) {
        return index;
      }
    }
  }
  return std::nullopt;
}

}  // namespace detail

std::optional<int32_t> SelectSubgroupMatrixConfig(
    const ComputeContextBase& context,
    bool is_fp16,
    std::initializer_list<SubgroupMatrixConfigPreference> preferences) {
  const auto candidates = GetSupportedSubgroupMatrixConfigIndices(context, is_fp16);
  return detail::SelectSubgroupMatrixConfigFromCandidates(candidates, preferences);
}

}  // namespace webgpu
}  // namespace onnxruntime
