// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_execution_provider_info.h"
#include "core/providers/cann/cann_provider_options.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_call.h"

namespace onnxruntime {
namespace cann {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "npu_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kMaxOpqueueNum = "max_opqueue_num";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
}  // namespace provider_option_names
}  // namespace cann

namespace {
const DeleteOnUnloadPtr<EnumNameMapping<ArenaExtendStrategy>> arena_extend_strategy_mapping =
    new EnumNameMapping<ArenaExtendStrategy>{
        {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
        {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
    };
}  // namespace

CANNExecutionProviderInfo CANNExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  CANNExecutionProviderInfo info{};
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              cann::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                uint32_t num_devices{};
                ORT_RETURN_IF_NOT(
                    CANN_CALL(aclrtGetDeviceCount(&num_devices)),
                    "aclrtGetDeviceCount() failed.");
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && (unsigned)info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddAssignmentToReference(cann::provider_option_names::kMaxOpqueueNum, info.max_opqueue_num)
          .AddAssignmentToReference(cann::provider_option_names::kMemLimit, info.npu_mem_limit)
          .AddAssignmentToEnumReference(
              cann::provider_option_names::kArenaExtendStrategy,
              *arena_extend_strategy_mapping, info.arena_extend_strategy)
          .AddAssignmentToReference(cann::provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .Parse(options));
  return info;
}

ProviderOptions CANNExecutionProviderInfo::ToProviderOptions(const CANNExecutionProviderInfo& info) {
  const ProviderOptions options{
      {cann::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cann::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.npu_mem_limit)},
      {cann::provider_option_names::kArenaExtendStrategy,
       EnumToName(*arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {cann::provider_option_names::kDoCopyInDefaultStream,
       MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {cann::provider_option_names::kMaxOpqueueNum, MakeStringWithClassicLocale(info.max_opqueue_num)}};
  return options;
}

ProviderOptions CANNExecutionProviderInfo::ToProviderOptions(const OrtCANNProviderOptions& info) {
  const ProviderOptions options{
      {cann::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cann::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.npu_mem_limit)},
      {cann::provider_option_names::kArenaExtendStrategy,
       EnumToName(*arena_extend_strategy_mapping, ArenaExtendStrategy(info.arena_extend_strategy))},
      {cann::provider_option_names::kDoCopyInDefaultStream,
       MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {cann::provider_option_names::kMaxOpqueueNum, MakeStringWithClassicLocale(info.max_opqueue_num)}};
  return options;
}
}  // namespace onnxruntime
