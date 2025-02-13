// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <string>

#include "core/framework/ortdevice.h"
#include "core/common/hash_combine.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

namespace migraphx_provider_option {
constexpr auto kDeviceId = "device_id";
constexpr auto kFp16Enable = "migraphx_fp16_enable";
constexpr auto kFp8Enable = "migraphx_fp8_enable";
constexpr auto kInt8Enable = "migraphx_int8_enable";
constexpr auto kInt8CalibTable = "migraphx_int8_calibration_table_name";
constexpr auto kInt8UseNativeCalibTable = "migraphx_int8_use_native_calibration_table";
constexpr auto kSaveCompiledModel = "migraphx_save_compiled_model";
constexpr auto kSaveModelPath = "migraphx_save_model_name";
constexpr auto kLoadCompiledModel = "migraphx_load_compiled_model";
constexpr auto kLoadModelPath = "migraphx_load_model_name";
constexpr auto kExhaustiveTune = "migraphx_exhaustive_tune";
constexpr auto kMemLimit = "migraphx_mem_limit";
constexpr auto kArenaExtendStrategy = "migraphx_arena_extend_strategy";
constexpr auto kGpuExternalAlloc = "migraphx_external_alloc";
constexpr auto kGpuExternalFree = "migraphx_external_free";
constexpr auto kGpuExternalEmptyCache = "migraphx_external_empty_cache";
}  // namespace migraphx_provider_option

// Information needed to construct MIGraphX execution providers.
struct MIGraphXExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};
  void* empty_cache{nullptr};

  MIGraphXExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
    empty_cache = nullptr;
  }

  MIGraphXExecutionProviderExternalAllocatorInfo(void* a, void* f, void* e) {
    alloc = a;
    free = f;
    empty_cache = e;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }
};

// Information needed to construct trt execution providers.
struct MIGraphXExecutionProviderInfo {
  std::string target_device;
  OrtDevice::DeviceId device_id{0};
  bool fp16_enable{false};
  bool fp8_enable{false};
  bool int8_enable{false};
  std::string int8_calibration_table_name{""};
  bool int8_use_native_calibration_table{false};
  bool save_compiled_model{true};
  std::string save_model_file{"./compiled_model.mxr"};
  bool load_compiled_model{true};
  std::string load_model_file{"./compiled_model.mxr"};
  bool exhaustive_tune{false};

  size_t mem_limit{std::numeric_limits<size_t>::max()};                             // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)

  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  MIGraphXExecutionProviderExternalAllocatorInfo external_allocator_info{};

  static MIGraphXExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const MIGraphXExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtMIGraphXProviderOptions& info);
};
}  // namespace onnxruntime

template <>
struct std::hash<::onnxruntime::MIGraphXExecutionProviderInfo> {
  size_t operator()(const ::onnxruntime::MIGraphXExecutionProviderInfo& info) const {
    size_t value{0xbc9f1d34};  // seed

    // Bits: device_id (16), arena_extend_strategy (reserved 2), boolean options (1 each)
    size_t data = static_cast<size_t>(info.device_id) ^
                  (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
                  (static_cast<size_t>(info.fp16_enable) << 18) ^
                  (static_cast<size_t>(info.int8_enable) << 19) ^
                  (static_cast<size_t>(info.int8_use_native_calibration_table) << 20) ^
                  (static_cast<size_t>(info.save_compiled_model) << 21) ^
                  (static_cast<size_t>(info.load_compiled_model) << 22) ^
                  (static_cast<size_t>(info.exhaustive_tune) << 23);
    onnxruntime::HashCombine(data, value);

    onnxruntime::HashCombine(info.mem_limit, value);

    // Memory pointers
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.alloc), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.free), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache), value);

    // The default memory arena cfg is not used in hashing right now.
    return value;
  }
};
