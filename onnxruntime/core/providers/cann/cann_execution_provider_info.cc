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
constexpr const char* kEnableCannGraph = "enable_cann_graph";
constexpr const char* kDumpGraphs = "dump_graphs";
constexpr const char* kPrecisionMode = "precision_mode";
constexpr const char* kOpSelectImplMode = "op_select_impl_mode";
constexpr const char* kOpTypeListForImplMode = "optypelist_for_implmode";
}  // namespace provider_option_names
}  // namespace cann

namespace {
const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
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
          .AddAssignmentToReference(cann::provider_option_names::kMemLimit, info.npu_mem_limit)
          .AddAssignmentToEnumReference(
              cann::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .AddAssignmentToReference(cann::provider_option_names::kEnableCannGraph, info.enable_cann_graph)
          .AddAssignmentToReference(cann::provider_option_names::kDumpGraphs, info.dump_graphs)
          .AddAssignmentToReference(cann::provider_option_names::kPrecisionMode, info.precision_mode)
          .AddAssignmentToReference(cann::provider_option_names::kOpSelectImplMode, info.op_select_impl_mode)
          .AddAssignmentToReference(cann::provider_option_names::kOpTypeListForImplMode, info.optypelist_for_implmode)
          .Parse(options));
  return info;
}

ProviderOptions CANNExecutionProviderInfo::ToProviderOptions(const CANNExecutionProviderInfo& info) {
  const ProviderOptions options{
      {cann::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cann::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.npu_mem_limit)},
      {cann::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {cann::provider_option_names::kEnableCannGraph, MakeStringWithClassicLocale(info.enable_cann_graph)},
      {cann::provider_option_names::kDumpGraphs, MakeStringWithClassicLocale(info.dump_graphs)},
      {cann::provider_option_names::kPrecisionMode, MakeStringWithClassicLocale(info.precision_mode)},
      {cann::provider_option_names::kOpSelectImplMode, MakeStringWithClassicLocale(info.op_select_impl_mode)},
      {cann::provider_option_names::kOpTypeListForImplMode, MakeStringWithClassicLocale(info.optypelist_for_implmode)}};
  return options;
}

ProviderOptions CANNExecutionProviderInfo::ToProviderOptions(const OrtCANNProviderOptions& info) {
  const ProviderOptions options{
      {cann::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cann::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.npu_mem_limit)},
      {cann::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, ArenaExtendStrategy(info.arena_extend_strategy))},
      {cann::provider_option_names::kEnableCannGraph, MakeStringWithClassicLocale(info.enable_cann_graph)},
      {cann::provider_option_names::kDumpGraphs, MakeStringWithClassicLocale(info.dump_graphs)},
      {cann::provider_option_names::kPrecisionMode, MakeStringWithClassicLocale(info.precision_mode)},
      {cann::provider_option_names::kOpSelectImplMode, MakeStringWithClassicLocale(info.op_select_impl_mode)},
      {cann::provider_option_names::kOpTypeListForImplMode, MakeStringWithClassicLocale(info.optypelist_for_implmode)}};
  return options;
}
}  // namespace onnxruntime
