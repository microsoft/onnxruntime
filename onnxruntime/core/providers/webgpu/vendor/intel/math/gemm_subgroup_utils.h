// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
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

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
