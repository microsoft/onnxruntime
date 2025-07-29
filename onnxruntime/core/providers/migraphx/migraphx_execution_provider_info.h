// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <limits>
#include <string>
#include <string_view>

#include "core/framework/ortdevice.h"
#include "core/common/hash_combine.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/provider_options.h"
#include "core/framework/provider_options_utils.h"
#include "core/session/onnxruntime_c_api.h"

using namespace std::literals::string_view_literals;

namespace onnxruntime {

namespace migraphx_provider_option {
constexpr auto kDeviceId = "device_id"sv;
constexpr auto kFp16Enable = "migraphx_fp16_enable"sv;
constexpr auto kBf16Enable = "migraphx_bf16_enable"sv;
constexpr auto kFp8Enable = "migraphx_fp8_enable"sv;
constexpr auto kInt8Enable = "migraphx_int8_enable"sv;
constexpr auto kInt8CalibTable = "migraphx_int8_calibration_table_name"sv;
constexpr auto kInt8UseNativeCalibTable = "migraphx_int8_use_native_calibration_table"sv;
constexpr auto kSaveCompiledModel = "migraphx_save_compiled_model"sv;
constexpr auto kSaveModelPath = "migraphx_save_model_name"sv;
constexpr auto kLoadCompiledModel = "migraphx_load_compiled_model"sv;
constexpr auto kLoadModelPath = "migraphx_load_model_name"sv;
constexpr auto kExhaustiveTune = "migraphx_exhaustive_tune"sv;
constexpr auto kMemLimit = "migraphx_mem_limit"sv;
constexpr auto kArenaExtendStrategy = "migraphx_arena_extend_strategy"sv;
constexpr auto kGpuExternalAlloc = "migraphx_external_alloc"sv;
constexpr auto kGpuExternalFree = "migraphx_external_free"sv;
constexpr auto kGpuExternalEmptyCache = "migraphx_external_empty_cache"sv;
constexpr auto kModelCacheDir = "migraphx_model_cache_dir"sv;
}  // namespace migraphx_provider_option

extern const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping;

// Information needed to construct trt execution providers.
struct MIGraphXExecutionProviderInfo {
  std::string target_device{"gpu"};
  OrtDevice::DeviceId device_id{0};
  bool fp16_enable{false};
  bool bf16_enable{false};
  bool fp8_enable{false};
  bool int8_enable{false};
  std::string int8_calibration_table_name{};
  bool int8_use_native_calibration_table{false};
  std::filesystem::path model_cache_dir{};
  bool exhaustive_tune{false};

  size_t mem_limit{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};

  OrtArenaCfg* default_memory_arena_cfg{nullptr};

  void* external_alloc{nullptr};
  void* external_free{nullptr};
  void* external_empty_cache{nullptr};

  bool UseExternalAlloc() const {
    return external_alloc != nullptr && external_free != nullptr;
  }

  MIGraphXExecutionProviderInfo() = default;

  explicit MIGraphXExecutionProviderInfo(const ProviderOptions& options);
  explicit MIGraphXExecutionProviderInfo(const OrtMIGraphXProviderOptions& options) noexcept;
  ProviderOptions ToProviderOptions() const;
};

}  // namespace onnxruntime

template <>
struct std::hash<::onnxruntime::MIGraphXExecutionProviderInfo> {
  size_t operator()(const ::onnxruntime::MIGraphXExecutionProviderInfo& info) const noexcept {
    size_t value{0xbc9f1d34};  // seed

    // Bits: device_id (16), arena_extend_strategy (reserved 2), boolean options (1 each)
    size_t data = static_cast<size_t>(info.device_id) ^
                  (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
                  (static_cast<size_t>(info.fp16_enable) << 18) ^
                  (static_cast<size_t>(info.int8_enable) << 19) ^
                  (static_cast<size_t>(info.int8_use_native_calibration_table) << 20) ^
                  (static_cast<size_t>(info.exhaustive_tune) << 21) ^
                  (static_cast<size_t>(info.bf16_enable) << 22);

    onnxruntime::HashCombine(data, value);

    onnxruntime::HashCombine(info.target_device, value);
    onnxruntime::HashCombine(info.default_memory_arena_cfg, value);
    onnxruntime::HashCombine(info.int8_calibration_table_name, value);
    onnxruntime::HashCombine(info.model_cache_dir, value);
    onnxruntime::HashCombine(info.mem_limit, value);

    // Memory pointers
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_alloc), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_free), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_empty_cache), value);

    // The default memory arena cfg is not used in hashing right now.
    return value;
  }
};
