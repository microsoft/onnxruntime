// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <limits>

#include "core/common/hash_combine.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// Information needed to construct ROCM execution providers.
struct ROCMExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};
  void* empty_cache{nullptr};

  ROCMExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
    empty_cache = nullptr;
  }

  ROCMExecutionProviderExternalAllocatorInfo(void* a, void* f, void* e) {
    alloc = a;
    free = f;
    empty_cache = e;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }
};

namespace rocm {
struct TunableOpInfo {
  bool enabled{false};
};
}  // namespace rocm

struct ROCMExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};                         // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  bool miopen_conv_exhaustive_search{false};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  // The following OrtArenaCfg instance only characterizes the behavior of the default memory
  // arena allocator and not any other auxiliary allocator that may also be part of the ROCM EP.
  // For example, auxiliary allocators `HIP_PINNED` and `HIP_CPU` will not be configured using this
  // arena config.
  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};
  bool miopen_conv_use_max_workspace{false};

  rocm::TunableOpInfo tunable_op{};

  static ROCMExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const ROCMExecutionProviderInfo& info);
};
}  // namespace onnxruntime

template<>
struct std::hash<::onnxruntime::rocm::TunableOpInfo> {
  size_t operator()(const ::onnxruntime::rocm::TunableOpInfo& info) const {
    size_t seed_and_value{0xbc9f1d34};
    onnxruntime::HashCombine(info.enabled, seed_and_value);
    return seed_and_value;
  }
};
