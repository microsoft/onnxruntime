// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_execution_provider_info.h"

#include "core/common/string_utils.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "hip_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
}  // namespace provider_option_names

ROCMExecutionProviderInfo ROCMExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  ROCMExecutionProviderInfo info{};

  // TODO validate info.device_id
  ReadProviderOption(options, provider_option_names::kDeviceId, info.device_id);
  ReadProviderOption(options, provider_option_names::kMemLimit, info.hip_mem_limit);
  ReadProviderOption(options, provider_option_names::kArenaExtendStrategy, info.arena_extend_strategy);

  return info;
}

ProviderOptions ROCMExecutionProviderInfo::ToProviderOptions(const ROCMExecutionProviderInfo& info) {
  const ProviderOptions options{
      {provider_option_names::kDeviceId, MakeString(info.device_id)},
      {provider_option_names::kMemLimit, MakeString(info.hip_mem_limit)},
      {provider_option_names::kArenaExtendStrategy, MakeString(info.arena_extend_strategy)},
  };

  return options;
}
}  // namespace onnxruntime
