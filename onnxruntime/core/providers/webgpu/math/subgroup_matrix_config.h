// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>

#include <gsl/span>

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {
namespace webgpu {

class ComputeContextBase;

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

// A subgroup-matrix configuration implemented by an operation, including any operation-specific metadata.
struct SubgroupMatrixConfig {
  wgpu::SubgroupMatrixComponentType componentType;
  wgpu::SubgroupMatrixComponentType resultComponentType;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t subgroupSize;
  bool needsPrepack;  // Whether input A needs layout optimization for subgroupMatrixLoad

  // True if this config's subgroup-matrix shape equals (m, n, k).
  constexpr bool Is(uint32_t m, uint32_t n, uint32_t k) const {
    return M == m && N == n && K == k;
  }
};

// A fixed-size adapter already guarantees the required size. An adapter exposing a range needs
// subgroup-size control so the kernel can select its required size instead of relying on the
// implementation's choice.
constexpr bool IsSubgroupSizeSupported(uint32_t adapter_min_size, uint32_t adapter_max_size,
                                       uint32_t required_size, bool has_subgroup_size_control) {
  return adapter_min_size <= required_size && required_size <= adapter_max_size &&
         (adapter_min_size == adapter_max_size || has_subgroup_size_control);
}

// Selects a subgroup-matrix configuration supported by both the operation and the device.
//
// `preferences` contains the complete configurations implemented by the operation, in
// performance-preference order. The first configuration also reported by the adapter and with a
// usable subgroup size is returned. Fixed-size adapters need no subgroup-size-control feature;
// adapters reporting a size range must support subgroup-size control.
//
// Returns the selected configuration by value, or `std::nullopt` when no configuration satisfies
// all requirements.
std::optional<SubgroupMatrixConfig> SelectSubgroupMatrixConfig(
    const ComputeContextBase& context,
    std::initializer_list<SubgroupMatrixConfig> preferences);

namespace detail {

// Separated from ComputeContext feature discovery for focused preference-order testing.
std::optional<SubgroupMatrixConfig> SelectSubgroupMatrixConfigFromAdapterConfigs(
    gsl::span<const wgpu::SubgroupMatrixConfig> adapter_configs,
    uint32_t adapter_min_subgroup_size,
    uint32_t adapter_max_subgroup_size,
    bool has_subgroup_size_control,
    std::initializer_list<SubgroupMatrixConfig> preferences);

}  // namespace detail

}  // namespace webgpu
}  // namespace onnxruntime
