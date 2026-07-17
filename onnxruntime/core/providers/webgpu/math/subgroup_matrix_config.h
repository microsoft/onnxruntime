// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {

// Human-readable names for wgpu::SubgroupMatrixComponentType, indexed by the enum value.
constexpr std::string_view ComponentTypeName[] = {"unknown", "f32", "f16", "u32", "i32"};
template <std::size_t N>
constexpr bool ValidateComponentTypeName(const std::array<wgpu::SubgroupMatrixComponentType, N>& component_type) {
  bool matched = true;
  for (auto type : component_type) {
    switch (type) {
      case wgpu::SubgroupMatrixComponentType::F32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F32)] == "f32";
        break;
      case wgpu::SubgroupMatrixComponentType::F16:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F16)] == "f16";
        break;
      case wgpu::SubgroupMatrixComponentType::U32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::U32)] == "u32";
        break;
      case wgpu::SubgroupMatrixComponentType::I32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::I32)] == "i32";
        break;
      default:
        return false;
    }

    if (!matched) {
      return matched;
    }
  }

  return matched;
}
static_assert(ValidateComponentTypeName<4>({wgpu::SubgroupMatrixComponentType::F32,
                                            wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::U32,
                                            wgpu::SubgroupMatrixComponentType::I32}),
              "The elements' sequence of ComponentTypeName array do not match wgpu::SubgroupMatrixComponentType");

// Vendor-agnostic subgroup matrix config: {componentType, resultComponentType, M, N, K,
// subgroupMinSize, subgroupMaxSize, needsPrepack}. Any GPU reporting a matching config from
// wgpu::AdapterPropertiesSubgroupMatrixConfigs is supported.
struct SupportedSubgroupMatrixConfig {
  wgpu::SubgroupMatrixComponentType componentType;
  wgpu::SubgroupMatrixComponentType resultComponentType;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t subgroupMinSize;
  uint32_t subgroupMaxSize;
  bool needsPrepack;  // Whether input A needs layout optimization for subgroupMatrixLoad

  // True if this config's subgroup-matrix shape equals (m, n, k).
  constexpr bool Is(uint32_t m, uint32_t n, uint32_t k) const {
    return M == m && N == n && K == k;
  }
};

// Subgroup matrix configs the subgroup-matrix kernels are implemented for.
inline constexpr std::array<SupportedSubgroupMatrixConfig, 4> supported_subgroup_matrix_configs = {{
    // 16x16x16 config with 128x128 tiles (NVIDIA Blackwell, subgroup size 32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 16, 16, 16, 32, 32, true},
    // 8x16x16 config (Intel Xe2/Xe3, subgroup size 16-32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 16, 16, 16, 32, true},
    // 8x8x8 config (Apple M-series, etc.)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 8, 8, 32, 32, false},
    {wgpu::SubgroupMatrixComponentType::F32, wgpu::SubgroupMatrixComponentType::F32, 8, 8, 8, 32, 32, false},
}};

// Returns true and sets config_index (into supported_subgroup_matrix_configs) when the device
// reports one of the supported configs matching the requested output precision. is_fp16 selects
// F16-output configs; otherwise F32-output configs.
bool IsSubgroupMatrixConfigSupported(const ComputeContextBase& context, bool is_fp16, int32_t& config_index);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
