// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace gpu_arch {
inline constexpr std::string_view kXeLpg = "xe-lpg";
inline constexpr std::string_view kXe3Lpg = "xe-3lpg";
}  // namespace gpu_arch

bool CanUseAVec4CooperativeLoad(std::string_view architecture,
                                uint32_t dim_inner,
                                int64_t elements_per_thread_y);

std::optional<uint32_t> SelectMatMulSubgroupSize(uint32_t adapter_min_subgroup_size,
                                                 uint32_t adapter_max_subgroup_size,
                                                 bool has_subgroup_size_control);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
