// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tensorrt/tensorrt_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace tensorrt {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasTrtOptions = "has_trt_options";
constexpr const char* kMaxWorkspaceSize = "trt_max_workspace_size";
constexpr const char* kFp16Enable = "trt_fp16_enable";
constexpr const char* kInt8Enable = "trt_int8_enable";
constexpr const char* kInt8CalibTable = "trt_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "trt_int8_use_native_calibration_table";
//constexpr const char* kForceSequentialEngineBuild = "trt_force_sequential_engine_build";
}  // namespace provider_option_names
}  // namespace tensorrt 

TensorrtExecutionProviderInfo TensorrtExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  TensorrtExecutionProviderInfo info{};
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              tensorrt::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                ORT_RETURN_IF_NOT(
                    CUDA_CALL(cudaGetDeviceCount(&num_devices)),
                    "cudaGetDeviceCount() failed.");
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kHasTrtOptions, info.has_trt_options)
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8CalibTable, info.int8_calibration_table_name)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8UseNativeCalibTable, info.int8_use_native_calibration_table)
          //.AddAssignmentToReference(tensorrt::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .Parse(options));

  return info;
}

ProviderOptions TensorrtExecutionProviderInfo::ToProviderOptions(const TensorrtExecutionProviderInfo& info) {
  const ProviderOptions options{
      {tensorrt::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {tensorrt::provider_option_names::kHasTrtOptions, MakeStringWithClassicLocale(info.has_trt_options)},
      {tensorrt::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {tensorrt::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {tensorrt::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {tensorrt::provider_option_names::kInt8CalibTable, MakeStringWithClassicLocale(info.int8_calibration_table_name)},
      {tensorrt::provider_option_names::kInt8UseNativeCalibTable, MakeStringWithClassicLocale(info.int8_use_native_calibration_table)},
      //{tensorrt::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.force_sequential_engine_build)},
  };

  return options;
}
}  // namespace onnxruntime
