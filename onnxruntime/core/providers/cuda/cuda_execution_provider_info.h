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
// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};
  void* empty_cache{nullptr};

  CUDAExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
    empty_cache = nullptr;
  }

  CUDAExecutionProviderExternalAllocatorInfo(void* a, void* f, void* e) {
    alloc = a;
    free = f;
    empty_cache = e;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }
};

namespace cuda {
struct TunableOpInfo {
  bool enable{false};
  bool tuning_enable{false};
  int max_tuning_duration_ms{};
};
}  // namespace cuda

struct CUDAExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};                         // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search{OrtCudnnConvAlgoSearchExhaustive};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  // The following OrtArenaCfg instance only characterizes the behavior of the default memory
  // arena allocator and not any other auxiliary allocator that may also be part of the CUDA EP.
  // For example, auxiliary allocators `CUDA_PINNED` and `CUDA_CPU` will not be configured using this
  // arena config.
  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};

  // By default, try to use as much as possible memory for algo search.
  // If set to false, use fix workspace size (32M) for Conv algo search, the final algo might not be the best.
  bool cudnn_conv_use_max_workspace{true};

  bool enable_cuda_graph{false};

  // By default, for Conv1D, will pad [N,C,D] to [N,C,D,1], if turn on, will pad to [N,C,1,D].
  bool cudnn_conv1d_pad_to_nc1d{false};

  cuda::TunableOpInfo tunable_op{};

  bool enable_skip_layer_norm_strict_mode{false};

  static CUDAExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const CUDAExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtCUDAProviderOptionsV2& info);
};
}  // namespace onnxruntime

template <>
struct std::hash<::onnxruntime::cuda::TunableOpInfo> {
  size_t operator()(const ::onnxruntime::cuda::TunableOpInfo& info) const {
    size_t seed_and_value{0xbc9f1d34};
    onnxruntime::HashCombine(info.enable, seed_and_value);
    onnxruntime::HashCombine(info.tuning_enable, seed_and_value);
    onnxruntime::HashCombine(info.max_tuning_duration_ms, seed_and_value);
    return seed_and_value;
  }
};
