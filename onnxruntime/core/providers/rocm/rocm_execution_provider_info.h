// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
namespace rocm {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "hip_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
}  // namespace provider_option_names
}  // namespace rocm

// Information needed to construct HIP execution providers.
struct ROCMExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t hip_mem_limit{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};

  static ROCMExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const ROCMExecutionProviderInfo& info);
};
}  // namespace onnxruntime
