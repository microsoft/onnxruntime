// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace rocm {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "hip_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kConvExhaustiveSearch = "conv_exhaustive_search";
}  // namespace provider_option_names
}  // namespace rocm

namespace {
const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};
}  // namespace

ROCMExecutionProviderInfo ROCMExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  ROCMExecutionProviderInfo info{};

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          // TODO validate info.device_id
          .AddAssignmentToReference(rocm::provider_option_names::kDeviceId, info.device_id)
          .AddAssignmentToReference(rocm::provider_option_names::kMemLimit, info.hip_mem_limit)
          .AddAssignmentToReference(rocm::provider_option_names::kConvExhaustiveSearch, info.miopen_conv_exhaustive_search)
          .AddAssignmentToEnumReference(
              rocm::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .Parse(options));

  return info;
}

ProviderOptions ROCMExecutionProviderInfo::ToProviderOptions(const ROCMExecutionProviderInfo& info) {
  const ProviderOptions options{
      {rocm::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {rocm::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.hip_mem_limit)},
      {rocm::provider_option_names::kConvExhaustiveSearch, MakeStringWithClassicLocale(info.miopen_conv_exhaustive_search)},
      {rocm::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
  };

  return options;
}
}  // namespace onnxruntime
