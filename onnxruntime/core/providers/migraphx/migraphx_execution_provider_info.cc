// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "migraphx_inc.h"
#include "migraphx_call.h"

namespace onnxruntime {

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};

namespace migraphx {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kFp16Enable = "trt_fp16_enable";
constexpr const char* kBf16Enable = "migx_bf16_enable";
constexpr const char* kFp8Enable = "migx_fp8_enable";
constexpr const char* kInt8Enable = "migx_int8_enable";
constexpr const char* kInt8CalibTable = "migx_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "migx_int8_use_native_calibration_table";
constexpr const char* kSaveCompiledModel = "migx_save_compiled_model";
constexpr const char* kSaveModelPath = "migx_save_model_name";
constexpr const char* kLoadCompiledModel = "migx_load_compiled_model";
constexpr const char* kLoadModelPath = "migx_load_model_name";
constexpr const char* kExhaustiveTune = "migx_exhaustive_tune";
constexpr const char* kMemLimit = "migx_mem_limit";
constexpr const char* kArenaExtendStrategy = "migx_arena_extend_strategy";
constexpr const char* kGpuExternalAlloc = "migx_external_alloc";
constexpr const char* kGpuExternalFree = "migx_external_free";
constexpr const char* kGpuExternalEmptyCache = "migx_external_empty_cache";

}  // namespace provider_option_names
}  // namespace migraphx

MIGraphXExecutionProviderInfo MIGraphXExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  MIGraphXExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;
  void* empty_cache = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              migraphx::provider_option_names::kDeviceId,
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
              migraphx::provider_option_names::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx::provider_option_names::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx::provider_option_names::kGpuExternalEmptyCache,
              [&empty_cache](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(migraphx::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kBf16Enable, info.bf16_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kFp8Enable, info.fp8_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kSaveCompiledModel, info.save_compiled_model)
          .AddAssignmentToReference(migraphx::provider_option_names::kLoadCompiledModel, info.load_compiled_model)
          .AddAssignmentToReference(migraphx::provider_option_names::kExhaustiveTune, info.exhaustive_tune)
          .AddAssignmentToReference(migraphx::provider_option_names::kMemLimit, info.mem_limit)
          .AddAssignmentToEnumReference(migraphx::provider_option_names::kArenaExtendStrategy, arena_extend_strategy_mapping, info.arena_extend_strategy)
          .Parse(options));

  MIGraphXExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;

  return info;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const MIGraphXExecutionProviderInfo& info) {
  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {migraphx::provider_option_names::kBf16Enable, MakeStringWithClassicLocale(info.bf16_enable)},
      {migraphx::provider_option_names::kFp8Enable, MakeStringWithClassicLocale(info.fp8_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {migraphx::provider_option_names::kSaveCompiledModel, MakeStringWithClassicLocale(info.save_compiled_model)},
      {migraphx::provider_option_names::kLoadCompiledModel, MakeStringWithClassicLocale(info.load_compiled_model)},
      {migraphx::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.mem_limit)},
      {migraphx::provider_option_names::kGpuExternalAlloc, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.alloc))},
      {migraphx::provider_option_names::kGpuExternalFree, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.free))},
      {migraphx::provider_option_names::kGpuExternalEmptyCache, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache))},
      {migraphx::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {migraphx::provider_option_names::kExhaustiveTune, MakeStringWithClassicLocale(info.exhaustive_tune)},
  };
  return options;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const OrtMIGraphXProviderOptions& info) {
  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.migraphx_fp16_enable)},
      {migraphx::provider_option_names::kBf16Enable, MakeStringWithClassicLocale(info.migraphx_bf16_enable)},
      {migraphx::provider_option_names::kFp8Enable, MakeStringWithClassicLocale(info.migraphx_fp8_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.migraphx_int8_enable)},
      {migraphx::provider_option_names::kSaveCompiledModel, MakeStringWithClassicLocale(info.migraphx_save_compiled_model)},
      {migraphx::provider_option_names::kLoadCompiledModel, MakeStringWithClassicLocale(info.migraphx_load_compiled_model)},
      {migraphx::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.migraphx_mem_limit)},
      {migraphx::provider_option_names::kArenaExtendStrategy, EnumToName(arena_extend_strategy_mapping, static_cast<onnxruntime::ArenaExtendStrategy>(info.migraphx_arena_extend_strategy))},
      {migraphx::provider_option_names::kExhaustiveTune, MakeStringWithClassicLocale(info.migraphx_exhaustive_tune)},
  };
  return options;
}
}  // namespace onnxruntime
