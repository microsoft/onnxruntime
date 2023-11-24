// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "gpu_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kMiopenConvExhaustiveSearch = "miopen_conv_exhaustive_search";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
constexpr const char* kGpuExternalAlloc = "gpu_external_alloc";
constexpr const char* kGpuExternalFree = "gpu_external_free";
constexpr const char* kGpuExternalEmptyCache = "gpu_external_empty_cache";
constexpr const char* kMiopenConvUseMaxWorkspace = "miopen_conv_use_max_workspace";
constexpr const char* kTunableOpEnable = "tunable_op_enable";
constexpr const char* kTunableOpTuningEnable = "tunable_op_tuning_enable";
constexpr const char* kTunableOpMaxTuningDurationMs = "tunable_op_max_tuning_duration_ms";
}  // namespace provider_option_names
}  // namespace rocm

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};

ROCMExecutionProviderInfo ROCMExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  ROCMExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;
  void* empty_cache = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              rocm::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                HIP_RETURN_IF_ERROR(hipGetDeviceCount(&num_devices));
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kGpuExternalEmptyCache,
              [&empty_cache](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(rocm::provider_option_names::kMemLimit, info.gpu_mem_limit)
          .AddAssignmentToEnumReference(
              rocm::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .AddAssignmentToReference(
              rocm::provider_option_names::kMiopenConvExhaustiveSearch,
              info.miopen_conv_exhaustive_search)
          .AddAssignmentToReference(rocm::provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .AddAssignmentToReference(rocm::provider_option_names::kMiopenConvUseMaxWorkspace, info.miopen_conv_use_max_workspace)
          .AddValueParser(
              rocm::provider_option_names::kTunableOpEnable,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.enable));
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kTunableOpTuningEnable,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.tuning_enable));
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kTunableOpMaxTuningDurationMs,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.max_tuning_duration_ms));
                return Status::OK();
              })
          .Parse(options));

  ROCMExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;
  return info;
}

ProviderOptions ROCMExecutionProviderInfo::ToProviderOptions(const ROCMExecutionProviderInfo& info) {
  const ProviderOptions options{
      {rocm::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {rocm::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
      {rocm::provider_option_names::kGpuExternalAlloc, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.alloc))},
      {rocm::provider_option_names::kGpuExternalFree, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.free))},
      {rocm::provider_option_names::kGpuExternalEmptyCache, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache))},
      {rocm::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {rocm::provider_option_names::kMiopenConvExhaustiveSearch, MakeStringWithClassicLocale(info.miopen_conv_exhaustive_search)},
      {rocm::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {rocm::provider_option_names::kMiopenConvUseMaxWorkspace, MakeStringWithClassicLocale(info.miopen_conv_use_max_workspace)},
      {rocm::provider_option_names::kTunableOpEnable, MakeStringWithClassicLocale(info.tunable_op.enable)},
      {rocm::provider_option_names::kTunableOpTuningEnable, MakeStringWithClassicLocale(info.tunable_op.tuning_enable)},
      {rocm::provider_option_names::kTunableOpMaxTuningDurationMs, MakeStringWithClassicLocale(info.tunable_op.max_tuning_duration_ms)},
  };

  return options;
}

ProviderOptions ROCMExecutionProviderInfo::ToProviderOptions(const OrtROCMProviderOptions& info) {
  const ProviderOptions options{
      {rocm::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {rocm::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
      {rocm::provider_option_names::kArenaExtendStrategy, EnumToName(arena_extend_strategy_mapping, static_cast<onnxruntime::ArenaExtendStrategy>(info.arena_extend_strategy))},
      {rocm::provider_option_names::kMiopenConvExhaustiveSearch, MakeStringWithClassicLocale(info.miopen_conv_exhaustive_search)},
      {rocm::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {rocm::provider_option_names::kTunableOpEnable, MakeStringWithClassicLocale(info.tunable_op_enable)},
      {rocm::provider_option_names::kTunableOpTuningEnable, MakeStringWithClassicLocale(info.tunable_op_tuning_enable)},
      {rocm::provider_option_names::kTunableOpMaxTuningDurationMs, MakeStringWithClassicLocale(info.tunable_op_max_tuning_duration_ms)},
  };

  return options;
}

}  // namespace onnxruntime
