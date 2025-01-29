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
constexpr const char* kFp8Enable = "migx_fp8_enable";
constexpr const char* kInt8Enable = "migx_int8_enable";
constexpr const char* kInt8CalibTable = "migx_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "migx_int8_use_native_calibration_table";
constexpr const char* kSaveCompiledModel = "migx_save_compiled_model";
constexpr const char* kSaveModelPath = "migx_save_model_name";
constexpr const char* kLoadCompiledModel = "migx_load_compiled_model";
constexpr const char* kLoadModelPath = "migx_load_model_name";
constexpr const char* kExhaustiveTune = "migx_exhaustive_tune";

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
          .AddAssignmentToReference(migraphx::provider_option_names::kFp8Enable, info.fp8_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(migraphx::provider_option_names::kSaveCompiledModel, info.save_compiled_model)
          .AddAssignmentToReference(migraphx::provider_option_names::kLoadCompiledModel, info.load_compiled_model)
          .AddAssignmentToReference(migraphx::provider_option_names::kExhaustiveTune, info.exhaustive_tune)
          .Parse(options));

  return info;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const MIGraphXExecutionProviderInfo& info) {
  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {migraphx::provider_option_names::kFp8Enable, MakeStringWithClassicLocale(info.fp8_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {migraphx::provider_option_names::kSaveCompiledModel, MakeStringWithClassicLocale(info.save_compiled_model)},
      {migraphx::provider_option_names::kLoadCompiledModel, MakeStringWithClassicLocale(info.load_compiled_model)},
      {migraphx::provider_option_names::kExhaustiveTune, MakeStringWithClassicLocale(info.exhaustive_tune)},
  };
  return options;
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions(const OrtMIGraphXProviderOptions& info) {
  const ProviderOptions options{
      {migraphx::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {migraphx::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.migraphx_fp16_enable)},
      {migraphx::provider_option_names::kFp8Enable, MakeStringWithClassicLocale(info.migraphx_fp8_enable)},
      {migraphx::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.migraphx_int8_enable)},
      {migraphx::provider_option_names::kSaveCompiledModel, MakeStringWithClassicLocale(info.migraphx_save_compiled_model)},
      {migraphx::provider_option_names::kLoadCompiledModel, MakeStringWithClassicLocale(info.migraphx_load_compiled_model)},
      {migraphx::provider_option_names::kExhaustiveTune, MakeStringWithClassicLocale(info.migraphx_exhaustive_tune)},
  };
  return options;
}
}  // namespace onnxruntime
