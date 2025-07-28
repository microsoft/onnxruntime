// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

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
          .AddAssignmentToReference(migraphx_provider_option::kBf16Enable, info.bf16_enable)
          .AddAssignmentToReference(migraphx_provider_option::kFp8Enable, info.fp8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kSaveCompiledModel, info.save_compiled_model)
          .AddAssignmentToReference(migraphx_provider_option::kLoadCompiledModel, info.load_compiled_model)
          .AddAssignmentToReference(migraphx_provider_option::kExhaustiveTune, info.exhaustive_tune)
          .AddAssignmentToReference(migraphx_provider_option::kMemLimit, info.mem_limit)
          .AddAssignmentToEnumReference(migraphx_provider_option::kArenaExtendStrategy, arena_extend_strategy_mapping, info.arena_extend_strategy)
          .Parse(options));

  MIGraphXExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;

  return info;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const MIGraphXExecutionProviderInfo& info) {
  const ProviderOptions options{
      {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(info.device_id)},
      {std::string{migraphx_provider_option::kFp16Enable}, MakeStringWithClassicLocale(info.fp16_enable)},
      {std::string{migraphx_provider_option::kBf16Enable}, MakeStringWithClassicLocale(info.bf16_enable)},
      {std::string{migraphx_provider_option::kFp8Enable}, MakeStringWithClassicLocale(info.fp8_enable)},
      {std::string{migraphx_provider_option::kInt8Enable}, MakeStringWithClassicLocale(info.int8_enable)},
      {std::string{migraphx_provider_option::kSaveCompiledModel}, MakeStringWithClassicLocale(info.save_compiled_model)},
      {std::string{migraphx_provider_option::kLoadCompiledModel}, MakeStringWithClassicLocale(info.load_compiled_model)},
      {std::string{migraphx_provider_option::kMemLimit}, MakeStringWithClassicLocale(info.mem_limit)},
      {std::string{migraphx_provider_option::kGpuExternalAlloc}, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.alloc))},
      {std::string{migraphx_provider_option::kGpuExternalFree}, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.free))},
      {std::string{migraphx_provider_option::kGpuExternalEmptyCache}, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache))},
      {std::string{migraphx_provider_option::kArenaExtendStrategy}, EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {std::string{migraphx_provider_option::kExhaustiveTune}, MakeStringWithClassicLocale(info.exhaustive_tune)},
  };
  return options;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const OrtMIGraphXProviderOptions& info) {
  const ProviderOptions options{
      {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(info.device_id)},
      {std::string{migraphx_provider_option::kFp16Enable}, MakeStringWithClassicLocale(info.migraphx_fp16_enable)},
      {std::string{migraphx_provider_option::kBf16Enable}, MakeStringWithClassicLocale(info.migraphx_bf16_enable)},
      {std::string{migraphx_provider_option::kFp8Enable}, MakeStringWithClassicLocale(info.migraphx_fp8_enable)},
      {std::string{migraphx_provider_option::kInt8Enable}, MakeStringWithClassicLocale(info.migraphx_int8_enable)},
      {std::string{migraphx_provider_option::kSaveCompiledModel}, MakeStringWithClassicLocale(info.migraphx_save_compiled_model)},
      {std::string{migraphx_provider_option::kLoadCompiledModel}, MakeStringWithClassicLocale(info.migraphx_load_compiled_model)},
      {std::string{migraphx_provider_option::kMemLimit}, MakeStringWithClassicLocale(info.migraphx_mem_limit)},
      {std::string{migraphx_provider_option::kArenaExtendStrategy}, EnumToName(arena_extend_strategy_mapping, static_cast<onnxruntime::ArenaExtendStrategy>(info.migraphx_arena_extend_strategy))},
      {std::string{migraphx_provider_option::kExhaustiveTune}, MakeStringWithClassicLocale(info.migraphx_exhaustive_tune)},
  };
  return options;
}
}  // namespace onnxruntime
