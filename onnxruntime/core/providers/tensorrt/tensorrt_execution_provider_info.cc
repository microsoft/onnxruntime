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
constexpr const char* kMaxPartitionIterations = "trt_max_partition_iterations";
constexpr const char* kMinSubgraphSize = "trt_min_subgraph_size";
constexpr const char* kMaxWorkspaceSize = "trt_max_workspace_size";
constexpr const char* kFp16Enable = "trt_fp16_enable";
constexpr const char* kInt8Enable = "trt_int8_enable";
constexpr const char* kInt8CalibTable = "trt_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "trt_int8_use_native_calibration_table";
constexpr const char* kDLAEnable = "trt_dla_enable";
constexpr const char* kDLACore = "trt_dla_core";
constexpr const char* kDumpSubgraphs = "trt_dump_subgraphs";
constexpr const char* kEngineCacheEnable = "trt_engine_cache_enable";
constexpr const char* kCachePath = "trt_engine_cache_path";
constexpr const char* kDecryptionEnable = "trt_engine_decryption_enable";
constexpr const char* kDecryptionLibPath = "trt_engine_decryption_lib_path";
constexpr const char* kForceSequentialEngineBuild = "trt_force_sequential_engine_build";
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
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxPartitionIterations, info.max_partition_iterations)
          .AddAssignmentToReference(tensorrt::provider_option_names::kMinSubgraphSize, info.min_subgraph_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8CalibTable, info.int8_calibration_table_name)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8UseNativeCalibTable, info.int8_use_native_calibration_table)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDLAEnable, info.dla_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDLACore, info.dla_core)		  
          .AddAssignmentToReference(tensorrt::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCacheEnable, info.engine_cache_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kCachePath, info.engine_cache_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionEnable, info.engine_decryption_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionLibPath, info.engine_decryption_lib_path) 
          .AddAssignmentToReference(tensorrt::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .Parse(options));

  return info;
}

ProviderOptions TensorrtExecutionProviderInfo::ToProviderOptions(const TensorrtExecutionProviderInfo& info) {

  const ProviderOptions options{
      {tensorrt::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {tensorrt::provider_option_names::kMaxPartitionIterations, MakeStringWithClassicLocale(info.max_partition_iterations)},
      {tensorrt::provider_option_names::kMinSubgraphSize, MakeStringWithClassicLocale(info.min_subgraph_size)},
      {tensorrt::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {tensorrt::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {tensorrt::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {tensorrt::provider_option_names::kInt8CalibTable, MakeStringWithClassicLocale(info.int8_calibration_table_name)},
      {tensorrt::provider_option_names::kInt8UseNativeCalibTable, MakeStringWithClassicLocale(info.int8_use_native_calibration_table)},
      {tensorrt::provider_option_names::kDLAEnable, MakeStringWithClassicLocale(info.dla_enable)},
      {tensorrt::provider_option_names::kDLACore, MakeStringWithClassicLocale(info.dla_core)},
      {tensorrt::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.dump_subgraphs)},
      {tensorrt::provider_option_names::kEngineCacheEnable, MakeStringWithClassicLocale(info.engine_cache_enable)},
      {tensorrt::provider_option_names::kCachePath, MakeStringWithClassicLocale(info.engine_cache_path)},
      {tensorrt::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.engine_decryption_enable)},
      {tensorrt::provider_option_names::kDecryptionLibPath, MakeStringWithClassicLocale(info.engine_decryption_lib_path)},
      {tensorrt::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.force_sequential_engine_build)},
  };
  return options;
}

ProviderOptions TensorrtExecutionProviderInfo::ToProviderOptions(const OrtTensorRTProviderOptions& info) {

  std::string kInt8CalibTable_ = "";
  if (info.trt_int8_calibration_table_name != nullptr)
    kInt8CalibTable_ = MakeStringWithClassicLocale(info.trt_int8_calibration_table_name);

  std::string kCachePath_ = "";
  if (info.trt_engine_cache_path != nullptr)
    kCachePath_ = MakeStringWithClassicLocale(info.trt_engine_cache_path);

  std::string kDecryptionLibPath_ = "";
  if (info.trt_engine_decryption_lib_path != nullptr)
    kDecryptionLibPath_ = MakeStringWithClassicLocale(info.trt_engine_decryption_lib_path);

  const ProviderOptions options{
      {tensorrt::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {tensorrt::provider_option_names::kMaxPartitionIterations, MakeStringWithClassicLocale(info.trt_max_partition_iterations)},
      {tensorrt::provider_option_names::kMinSubgraphSize, MakeStringWithClassicLocale(info.trt_min_subgraph_size)},
      {tensorrt::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.trt_max_workspace_size)},
      {tensorrt::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.trt_fp16_enable)},
      {tensorrt::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.trt_int8_enable)},
      {tensorrt::provider_option_names::kInt8CalibTable, kInt8CalibTable_},
      {tensorrt::provider_option_names::kInt8UseNativeCalibTable, MakeStringWithClassicLocale(info.trt_int8_use_native_calibration_table)},
      {tensorrt::provider_option_names::kDLAEnable, MakeStringWithClassicLocale(info.trt_dla_enable)},
      {tensorrt::provider_option_names::kDLACore, MakeStringWithClassicLocale(info.trt_dla_core)},
      {tensorrt::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.trt_dump_subgraphs)},
      {tensorrt::provider_option_names::kEngineCacheEnable, MakeStringWithClassicLocale(info.trt_engine_cache_enable)},
      {tensorrt::provider_option_names::kCachePath, kCachePath_},
      {tensorrt::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.trt_engine_decryption_enable)},
      {tensorrt::provider_option_names::kDecryptionLibPath, kDecryptionLibPath_},
      {tensorrt::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.trt_force_sequential_engine_build)},
  };
  return options;
}
}  // namespace onnxruntime
