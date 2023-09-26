// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/migraphx/migraphx_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "migraphx_inc.h"
#include "migraphx_call.h"

namespace onnxruntime {
namespace migraphx {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kFp16Enable = "trt_fp16_enable";
constexpr const char* kInt8Enable = "migx_int8_enable";
constexpr const char* kInt8CalibTable = "migx_int8_calibration_table_name";

}  // namespace provider_option_names
}  // namespace migraphx

MIGraphXExecutionProviderInfo MIGraphXExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  MIGraphXExecutionProviderInfo info{};
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
          .AddAssignmentToReference(migraphx::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kInt8Enable, info.int8_enable)
          .Parse(options));

  return info;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const MIGraphXExecutionProviderInfo& info) {
  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
  };
  return options;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const OrtMIGraphXProviderOptions& info) {

  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.migraphx_fp16_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.migraphx_int8_enable)},
  };
  return options;
}
}  // namespace onnxruntime
