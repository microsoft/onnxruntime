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
  size_t hash() const{
    return static_cast<size_t>(device_id) ^
         gpu_mem_limit ^
         (static_cast<size_t>(arena_extend_strategy) << 16) ^
         (static_cast<size_t>(miopen_conv_exhaustive_search) << 18) ^
         (static_cast<size_t>(do_copy_in_default_stream) << 20) ^
         (static_cast<size_t>(has_user_compute_stream) << 22);
  }

  static ROCMExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const ROCMExecutionProviderInfo& info);
};
}  // namespace onnxruntime
