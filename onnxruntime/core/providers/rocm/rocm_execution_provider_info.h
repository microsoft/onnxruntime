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
  bool enable{false};
  bool tuning_enable{false};
  int max_tuning_duration_ms{};
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

  // By default, try to use as much as possible memory for algo search.
  // If set to false, use fix workspace size (32M) for Conv algo search, the final algo might not be the best.
  bool miopen_conv_use_max_workspace{true};

  bool enable_hip_graph{false};

  rocm::TunableOpInfo tunable_op{};

  static ROCMExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const ROCMExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtROCMProviderOptions& info);
};
}  // namespace onnxruntime

template <>
struct std::hash<::onnxruntime::ROCMExecutionProviderInfo> {
  size_t operator()(const ::onnxruntime::ROCMExecutionProviderInfo& info) const {
    size_t value{0xbc9f1d34};  // seed

    // Bits: device_id (16), arena_extend_strategy/miopen_conv_exhaustive_search (reserved 2), boolean options (1 each)
    size_t data = static_cast<size_t>(info.device_id) ^
                  (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
                  (static_cast<size_t>(info.miopen_conv_exhaustive_search) << 18) ^
                  (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
                  (static_cast<size_t>(info.has_user_compute_stream) << 21) ^
                  (static_cast<size_t>(info.miopen_conv_use_max_workspace) << 22) ^
                  (static_cast<size_t>(info.enable_hip_graph) << 23) ^
                  (static_cast<size_t>(info.tunable_op.enable) << 24) ^
                  (static_cast<size_t>(info.tunable_op.tuning_enable) << 25);
    onnxruntime::HashCombine(data, value);

    onnxruntime::HashCombine(info.gpu_mem_limit, value);
    onnxruntime::HashCombine(info.tunable_op.max_tuning_duration_ms, value);

    // Memory pointers
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.user_compute_stream), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.alloc), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.free), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache), value);

    // The default memory arena cfg is not used in hashing right now.
    return value;
  }
};
