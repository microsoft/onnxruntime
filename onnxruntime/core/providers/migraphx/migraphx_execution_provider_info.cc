// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/migraphx/migraphx_inc.h"
#include "core/providers/migraphx/migraphx_call.h"

namespace onnxruntime {

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};

MIGraphXExecutionProviderInfo MIGraphXExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  MIGraphXExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;
  void* empty_cache = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              migraphx_provider_option::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                ORT_RETURN_IF_ERROR(HIP_CALL(hipGetDeviceCount(&num_devices)));
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalEmptyCache,
              [&empty_cache](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(migraphx_provider_option::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(migraphx_provider_option::kFp8Enable, info.fp8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kFp8Enable, info.fp8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kModelCacheDir, info.model_cache_dir)
          .AddAssignmentToReference(migraphx_provider_option::kExhaustiveTune, info.exhaustive_tune)
          .AddAssignmentToReference(migraphx_provider_option::kMemLimit, info.mem_limit)
          .AddAssignmentToEnumReference(migraphx_provider_option::kArenaExtendStrategy, arena_extend_strategy_mapping, info.arena_extend_strategy)
          .Parse(options));

  MIGraphXExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;

  return info;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const MIGraphXExecutionProviderInfo& info) {
  return {
      {migraphx_provider_option::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx_provider_option::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {migraphx_provider_option::kFp8Enable, MakeStringWithClassicLocale(info.fp8_enable)},
      {migraphx_provider_option::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {migraphx_provider_option::kModelCacheDir, MakeStringWithClassicLocale(info.model_cache_dir)},
      {migraphx_provider_option::kMemLimit, MakeStringWithClassicLocale(info.mem_limit)},
      {migraphx_provider_option::kGpuExternalAlloc, MakeStringWithClassicLocale(info.external_allocator_info.alloc)},
      {migraphx_provider_option::kGpuExternalFree, MakeStringWithClassicLocale(info.external_allocator_info.free)},
      {migraphx_provider_option::kGpuExternalEmptyCache, MakeStringWithClassicLocale(info.external_allocator_info.empty_cache)},
      {migraphx_provider_option::kArenaExtendStrategy, EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {migraphx_provider_option::kExhaustiveTune, MakeStringWithClassicLocale(info.exhaustive_tune)}
  };
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const OrtMIGraphXProviderOptions& info) {
  return {
      {migraphx_provider_option::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx_provider_option::kFp16Enable, MakeStringWithClassicLocale(info.migraphx_fp16_enable)},
      {migraphx_provider_option::kFp8Enable, MakeStringWithClassicLocale(info.migraphx_fp8_enable)},
      {migraphx_provider_option::kInt8Enable, MakeStringWithClassicLocale(info.migraphx_int8_enable)},
      {migraphx_provider_option::kModelCacheDir, MakeStringWithClassicLocale(info.migraphx_cache_dir)},
      {migraphx_provider_option::kMemLimit, MakeStringWithClassicLocale(info.migraphx_mem_limit)},
      {migraphx_provider_option::kArenaExtendStrategy, EnumToName(arena_extend_strategy_mapping, static_cast<onnxruntime::ArenaExtendStrategy>(info.migraphx_arena_extend_strategy))},
      {migraphx_provider_option::kExhaustiveTune, MakeStringWithClassicLocale(info.migraphx_exhaustive_tune)}
  };
}

}  // namespace onnxruntime
