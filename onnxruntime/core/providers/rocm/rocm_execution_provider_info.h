// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
// Information needed to construct HIP execution providers.
struct ROCMExecutionProviderExternalAllocatorInfo {
  const void* alloc{nullptr};
  const void* free{nullptr};

  ROCMExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
  }

  ROCMExecutionProviderExternalAllocatorInfo(void* a, void* f) {
    alloc = a;
    free = f;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }

  bool operator==(const ROCMExecutionProviderExternalAllocatorInfo& other) const {
    return alloc == other.alloc && free == other.free;
  }
};

struct ROCMExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};
  bool miopen_conv_exhaustive_search{false};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};

  static ROCMExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const ROCMExecutionProviderInfo& info);

  bool operator==(const ROCMExecutionProviderInfo& other) const;
};
}  // namespace onnxruntime

namespace std {
using onnxruntime::ROCMExecutionProviderInfo;
template <>
struct hash<ROCMExecutionProviderInfo> {
  size_t operator()(const ROCMExecutionProviderInfo& info) const {
    return static_cast<size_t>(info.device_id) ^
           info.gpu_mem_limit ^
           (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
           (static_cast<size_t>(info.miopen_conv_exhaustive_search) << 18) ^
           (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
           (static_cast<size_t>(info.has_user_compute_stream) << 22);
  }
};
}  // namespace std