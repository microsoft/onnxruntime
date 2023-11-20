// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tensorrt/tensorrt_execution_provider_info.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace tensorrt {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";
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
constexpr const char* kEngineCachePath = "trt_engine_cache_path";
constexpr const char* kEngineCachePrefix = "trt_engine_cache_prefix";
constexpr const char* kDecryptionEnable = "trt_engine_decryption_enable";
constexpr const char* kDecryptionLibPath = "trt_engine_decryption_lib_path";
constexpr const char* kForceSequentialEngineBuild = "trt_force_sequential_engine_build";
// add new provider option name here.
constexpr const char* kContextMemorySharingEnable = "trt_context_memory_sharing_enable";
constexpr const char* kLayerNormFP32Fallback = "trt_layer_norm_fp32_fallback";
constexpr const char* kTimingCacheEnable = "trt_timing_cache_enable";
constexpr const char* kTimingCachePath = "trt_timing_cache_path";
constexpr const char* kForceTimingCacheMatch = "trt_force_timing_cache";
constexpr const char* kDetailedBuildLog = "trt_detailed_build_log";
constexpr const char* kBuildHeuristics = "trt_build_heuristics_enable";
constexpr const char* kSparsityEnable = "trt_sparsity_enable";
constexpr const char* kBuilderOptimizationLevel = "trt_builder_optimization_level";
constexpr const char* kAuxiliaryStreams = "trt_auxiliary_streams";
constexpr const char* kTacticSources = "trt_tactic_sources";
constexpr const char* kExtraPluginLibPaths = "trt_extra_plugin_lib_paths";
constexpr const char* kProfilesMinShapes = "trt_profile_min_shapes";
constexpr const char* kProfilesMaxShapes = "trt_profile_max_shapes";
constexpr const char* kProfilesOptShapes = "trt_profile_opt_shapes";
constexpr const char* kCudaGraphEnable = "trt_cuda_graph_enable";
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
                CUDA_RETURN_IF_ERROR(cudaGetDeviceCount(&num_devices));
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxPartitionIterations, info.max_partition_iterations)
          .AddAssignmentToReference(tensorrt::provider_option_names::kHasUserComputeStream, info.has_user_compute_stream)
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
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCachePath, info.engine_cache_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCachePrefix, info.engine_cache_prefix)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionEnable, info.engine_decryption_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionLibPath, info.engine_decryption_lib_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .AddAssignmentToReference(tensorrt::provider_option_names::kContextMemorySharingEnable, info.context_memory_sharing_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kLayerNormFP32Fallback, info.layer_norm_fp32_fallback)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTimingCacheEnable, info.timing_cache_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTimingCachePath, info.timing_cache_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kForceTimingCacheMatch, info.force_timing_cache)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
          .AddAssignmentToReference(tensorrt::provider_option_names::kBuildHeuristics, info.build_heuristics_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kSparsityEnable, info.sparsity_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kBuilderOptimizationLevel, info.builder_optimization_level)
          .AddAssignmentToReference(tensorrt::provider_option_names::kAuxiliaryStreams, info.auxiliary_streams)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTacticSources, info.tactic_sources)
          .AddAssignmentToReference(tensorrt::provider_option_names::kExtraPluginLibPaths, info.extra_plugin_lib_paths)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
          .Parse(options));  // add new provider option here.

  return info;
}

ProviderOptions TensorrtExecutionProviderInfo::ToProviderOptions(const TensorrtExecutionProviderInfo& info) {
  const ProviderOptions options{
      {tensorrt::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {tensorrt::provider_option_names::kMaxPartitionIterations, MakeStringWithClassicLocale(info.max_partition_iterations)},
      {tensorrt::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
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
      {tensorrt::provider_option_names::kEngineCachePath, MakeStringWithClassicLocale(info.engine_cache_path)},
      {tensorrt::provider_option_names::kEngineCachePrefix, MakeStringWithClassicLocale(info.engine_cache_prefix)},
      {tensorrt::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.engine_decryption_enable)},
      {tensorrt::provider_option_names::kDecryptionLibPath, MakeStringWithClassicLocale(info.engine_decryption_lib_path)},
      {tensorrt::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.force_sequential_engine_build)},
      // add new provider option here.
      {tensorrt::provider_option_names::kContextMemorySharingEnable, MakeStringWithClassicLocale(info.context_memory_sharing_enable)},
      {tensorrt::provider_option_names::kLayerNormFP32Fallback, MakeStringWithClassicLocale(info.layer_norm_fp32_fallback)},
      {tensorrt::provider_option_names::kTimingCacheEnable, MakeStringWithClassicLocale(info.timing_cache_enable)},
      {tensorrt::provider_option_names::kTimingCachePath, MakeStringWithClassicLocale(info.timing_cache_path)},
      {tensorrt::provider_option_names::kForceTimingCacheMatch, MakeStringWithClassicLocale(info.force_timing_cache)},
      {tensorrt::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.detailed_build_log)},
      {tensorrt::provider_option_names::kBuildHeuristics, MakeStringWithClassicLocale(info.build_heuristics_enable)},
      {tensorrt::provider_option_names::kSparsityEnable, MakeStringWithClassicLocale(info.sparsity_enable)},
      {tensorrt::provider_option_names::kBuilderOptimizationLevel, MakeStringWithClassicLocale(info.builder_optimization_level)},
      {tensorrt::provider_option_names::kAuxiliaryStreams, MakeStringWithClassicLocale(info.auxiliary_streams)},
      {tensorrt::provider_option_names::kTacticSources, MakeStringWithClassicLocale(info.tactic_sources)},
      {tensorrt::provider_option_names::kExtraPluginLibPaths, MakeStringWithClassicLocale(info.extra_plugin_lib_paths)},
      {tensorrt::provider_option_names::kProfilesMinShapes, MakeStringWithClassicLocale(info.profile_min_shapes)},
      {tensorrt::provider_option_names::kProfilesMaxShapes, MakeStringWithClassicLocale(info.profile_max_shapes)},
      {tensorrt::provider_option_names::kProfilesOptShapes, MakeStringWithClassicLocale(info.profile_opt_shapes)},
      {tensorrt::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.cuda_graph_enable)},
  };
  return options;
}

ProviderOptions TensorrtExecutionProviderInfo::ToProviderOptions(const OrtTensorRTProviderOptionsV2& info) {
  auto empty_if_null = [](const char* s) { return s != nullptr ? std::string{s} : std::string{}; };
  const std::string kInt8CalibTable_ = empty_if_null(info.trt_int8_calibration_table_name);
  const std::string kEngineCachePath_ = empty_if_null(info.trt_engine_cache_path);
  const std::string kEngineCachePrefix_ = empty_if_null(info.trt_engine_cache_prefix);
  const std::string kTimingCachePath_ = empty_if_null(info.trt_timing_cache_path);
  const std::string kTacticSources_ = empty_if_null(info.trt_tactic_sources);
  const std::string kDecryptionLibPath_ = empty_if_null(info.trt_engine_decryption_lib_path);
  const std::string kExtraPluginLibPaths_ = empty_if_null(info.trt_extra_plugin_lib_paths);
  const std::string kProfilesMinShapes_ = empty_if_null(info.trt_profile_min_shapes);
  const std::string kProfilesMaxShapes_ = empty_if_null(info.trt_profile_max_shapes);
  const std::string kProfilesOptShapes_ = empty_if_null(info.trt_profile_opt_shapes);

  const ProviderOptions options{
      {tensorrt::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {tensorrt::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
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
      {tensorrt::provider_option_names::kEngineCachePath, kEngineCachePath_},
      {tensorrt::provider_option_names::kEngineCachePrefix, kEngineCachePrefix_},
      {tensorrt::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.trt_engine_decryption_enable)},
      {tensorrt::provider_option_names::kDecryptionLibPath, kDecryptionLibPath_},
      {tensorrt::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.trt_force_sequential_engine_build)},
      {tensorrt::provider_option_names::kContextMemorySharingEnable, MakeStringWithClassicLocale(info.trt_context_memory_sharing_enable)},
      {tensorrt::provider_option_names::kLayerNormFP32Fallback, MakeStringWithClassicLocale(info.trt_layer_norm_fp32_fallback)},
      {tensorrt::provider_option_names::kTimingCacheEnable, MakeStringWithClassicLocale(info.trt_timing_cache_enable)},
      {tensorrt::provider_option_names::kTimingCachePath, kTimingCachePath_},
      {tensorrt::provider_option_names::kForceTimingCacheMatch, MakeStringWithClassicLocale(info.trt_force_timing_cache)},
      {tensorrt::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.trt_detailed_build_log)},
      {tensorrt::provider_option_names::kBuildHeuristics, MakeStringWithClassicLocale(info.trt_build_heuristics_enable)},
      {tensorrt::provider_option_names::kSparsityEnable, MakeStringWithClassicLocale(info.trt_sparsity_enable)},
      {tensorrt::provider_option_names::kBuilderOptimizationLevel, MakeStringWithClassicLocale(info.trt_builder_optimization_level)},
      {tensorrt::provider_option_names::kAuxiliaryStreams, MakeStringWithClassicLocale(info.trt_auxiliary_streams)},
      {tensorrt::provider_option_names::kTacticSources, kTacticSources_},
      {tensorrt::provider_option_names::kExtraPluginLibPaths, kExtraPluginLibPaths_},
      {tensorrt::provider_option_names::kProfilesMinShapes, kProfilesMinShapes_},
      {tensorrt::provider_option_names::kProfilesMaxShapes, kProfilesMaxShapes_},
      {tensorrt::provider_option_names::kProfilesOptShapes, kProfilesOptShapes_},
      {tensorrt::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.trt_cuda_graph_enable)},
  };
  return options;
}

/**
 * Update OrtTensorRTProviderOptionsV2 instance with ProviderOptions (map of string-based key-value pairs)
 *
 * Please note that it will reset the OrtTensorRTProviderOptionsV2 instance first and then set up the provided provider options
 * See TensorrtExecutionProviderInfo::FromProviderOptions() for more details. This function will be called by the C API UpdateTensorRTProviderOptions() also.
 *
 * \param provider_options - a pointer to OrtTensorRTProviderOptionsV2 instance
 * \param options - a reference to ProviderOptions instance
 * \param string_copy - if it's true, it uses strncpy() to copy 'provider option' string from ProviderOptions instance to where the 'provider option' const char pointer in OrtTensorRTProviderOptionsV2 instance points to.
 *                      it it's false, it only saves the pointer and no strncpy().
 *
 * Note: If there is strncpy involved, please remember to deallocate or simply call C API ReleaseTensorRTProviderOptions.
 */
void TensorrtExecutionProviderInfo::UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) {
  if (provider_options == nullptr) {
    return;
  }
  auto copy_string_if_needed = [&](std::string& s_in) {
    if (string_copy) {
      char* dest = nullptr;
      auto str_size = s_in.size();
      if (str_size == 0) {
        return (const char*)nullptr;
      } else {
        dest = new char[str_size + 1];
#ifdef _MSC_VER
        strncpy_s(dest, str_size + 1, s_in.c_str(), str_size);
#else
        strncpy(dest, s_in.c_str(), str_size);
#endif
        dest[str_size] = '\0';
        return (const char*)dest;
      }
    } else {
      return s_in.c_str();
    }
  };

  TensorrtExecutionProviderInfo internal_options = onnxruntime::TensorrtExecutionProviderInfo::FromProviderOptions(options);
  auto& trt_provider_options_v2 = *reinterpret_cast<OrtTensorRTProviderOptionsV2*>(provider_options);
  trt_provider_options_v2.device_id = internal_options.device_id;

  // The 'has_user_compute_stream' of the OrtTensorRTProviderOptionsV2 instance can be set by C API UpdateTensorRTProviderOptionsWithValue() as well
  // We only set the 'has_user_compute_stream' of the OrtTensorRTProviderOptionsV2 instance if it is provided in options
  if (options.find("has_user_compute_stream") != options.end()) {
    trt_provider_options_v2.has_user_compute_stream = internal_options.has_user_compute_stream;
  }

  trt_provider_options_v2.trt_max_partition_iterations = internal_options.max_partition_iterations;
  trt_provider_options_v2.trt_min_subgraph_size = internal_options.min_subgraph_size;
  trt_provider_options_v2.trt_max_workspace_size = internal_options.max_workspace_size;
  trt_provider_options_v2.trt_fp16_enable = internal_options.fp16_enable;
  trt_provider_options_v2.trt_int8_enable = internal_options.int8_enable;

  trt_provider_options_v2.trt_int8_calibration_table_name = copy_string_if_needed(internal_options.int8_calibration_table_name);

  trt_provider_options_v2.trt_int8_use_native_calibration_table = internal_options.int8_use_native_calibration_table;
  trt_provider_options_v2.trt_dla_enable = internal_options.dla_enable;
  trt_provider_options_v2.trt_dla_core = internal_options.dla_core;
  trt_provider_options_v2.trt_dump_subgraphs = internal_options.dump_subgraphs;
  trt_provider_options_v2.trt_engine_cache_enable = internal_options.engine_cache_enable;

  trt_provider_options_v2.trt_engine_cache_path = copy_string_if_needed(internal_options.engine_cache_path);
  trt_provider_options_v2.trt_engine_cache_prefix = copy_string_if_needed(internal_options.trt_engine_cache_prefix);
  trt_provider_options_v2.trt_timing_cache_path = copy_string_if_needed(internal_options.timing_cache_path);

  trt_provider_options_v2.trt_engine_decryption_enable = internal_options.engine_decryption_enable;

  trt_provider_options_v2.trt_engine_decryption_lib_path = copy_string_if_needed(internal_options.engine_decryption_lib_path);

  trt_provider_options_v2.trt_force_sequential_engine_build = internal_options.force_sequential_engine_build;
  trt_provider_options_v2.trt_context_memory_sharing_enable = internal_options.context_memory_sharing_enable;
  trt_provider_options_v2.trt_layer_norm_fp32_fallback = internal_options.layer_norm_fp32_fallback;
  trt_provider_options_v2.trt_timing_cache_enable = internal_options.timing_cache_enable;
  trt_provider_options_v2.trt_force_timing_cache = internal_options.force_timing_cache;
  trt_provider_options_v2.trt_detailed_build_log = internal_options.detailed_build_log;
  trt_provider_options_v2.trt_build_heuristics_enable = internal_options.build_heuristics_enable;
  trt_provider_options_v2.trt_sparsity_enable = internal_options.sparsity_enable;
  trt_provider_options_v2.trt_builder_optimization_level = internal_options.builder_optimization_level;
  trt_provider_options_v2.trt_auxiliary_streams = internal_options.auxiliary_streams;

  trt_provider_options_v2.trt_tactic_sources = copy_string_if_needed(internal_options.tactic_sources);
  trt_provider_options_v2.trt_extra_plugin_lib_paths = copy_string_if_needed(internal_options.extra_plugin_lib_paths);
  trt_provider_options_v2.trt_profile_min_shapes = copy_string_if_needed(internal_options.profile_min_shapes);
  trt_provider_options_v2.trt_profile_max_shapes = copy_string_if_needed(internal_options.profile_max_shapes);
  trt_provider_options_v2.trt_profile_opt_shapes = copy_string_if_needed(internal_options.profile_opt_shapes);

  trt_provider_options_v2.trt_cuda_graph_enable = internal_options.cuda_graph_enable;
}
}  // namespace onnxruntime
