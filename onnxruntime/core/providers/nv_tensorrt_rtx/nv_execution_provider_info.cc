// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_info.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
NvExecutionProviderInfo NvExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options,
                                                                     const ConfigOptions& session_options) {
  NvExecutionProviderInfo info{};
  void* user_compute_stream = nullptr;
  void* onnx_bytestream = nullptr;
  void* external_data_bytestream = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              nv::provider_option_names::kDeviceId,
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
          .AddAssignmentToReference(nv::provider_option_names::kHasUserComputeStream, info.has_user_compute_stream)
          .AddValueParser(
              nv::provider_option_names::kUserComputeStream,
              [&user_compute_stream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                user_compute_stream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
          .AddAssignmentToReference(nv::provider_option_names::kMaxSharedMemSize, info.max_shared_mem_size)
          .AddAssignmentToReference(nv::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCacheEnable, info.engine_cache_enable)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCachePath, info.engine_cache_path)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCachePrefix, info.engine_cache_prefix)
          .AddAssignmentToReference(nv::provider_option_names::kEngineDecryptionEnable, info.engine_decryption_enable)
          .AddAssignmentToReference(nv::provider_option_names::kEngineDecryptionLibPath, info.engine_decryption_lib_path)
          .AddAssignmentToReference(nv::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .AddAssignmentToReference(nv::provider_option_names::kTimingCacheEnable, info.timing_cache_enable)
          .AddAssignmentToReference(nv::provider_option_names::kTimingCachePath, info.timing_cache_path)
          .AddAssignmentToReference(nv::provider_option_names::kForceTimingCacheMatch, info.force_timing_cache)
          .AddAssignmentToReference(nv::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
          .AddAssignmentToReference(nv::provider_option_names::kBuilderOptimizationLevel, info.builder_optimization_level)
          .AddAssignmentToReference(nv::provider_option_names::kUseExternalDataInitializer, info.use_external_data_initializer)
          .AddAssignmentToReference(nv::provider_option_names::kMultiProfileEnable, info.multi_profile_enable)
          .AddAssignmentToReference(nv::provider_option_names::kWeightStrippedEngineEnable, info.weight_stripped_engine_enable)
          .AddAssignmentToReference(nv::provider_option_names::kOnnxModelFolderPath, info.onnx_model_folder_path)
          .AddValueParser(
              nv::provider_option_names::kONNXBytestream,
              [&onnx_bytestream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                onnx_bytestream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kONNXBytestreamSize, info.onnx_bytestream_size)
          .AddValueParser(
              nv::provider_option_names::kExternalDataBytestream,
              [&external_data_bytestream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                external_data_bytestream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kExternalDataBytestreamSize, info.external_data_bytestream_size)
          .Parse(options));  // add new provider option here.

  info.user_compute_stream = user_compute_stream;
  info.has_user_compute_stream = (user_compute_stream != nullptr);
  info.onnx_bytestream = onnx_bytestream;
  info.external_data_bytestream = external_data_bytestream;

  // EP context settings
  // when EP context is enabled, default is to embed the engine in the context model
  // weight stripped engine is always enabled when EP context is enabled

  const auto ep_context_enable = session_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0");
  if (ep_context_enable == "0") {
    info.dump_ep_context_model = false;
  } else if (ep_context_enable == "1") {
    info.dump_ep_context_model = true;
    // We want to reenable weightless engines as soon constant initializers are supported as inputs
    info.weight_stripped_engine_enable = false;
  } else {
    ORT_THROW("Invalid ", kOrtSessionOptionEpContextEnable, " must 0 or 1");
  }
  info.ep_context_file_path = session_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");

  // If embed mode is not specified, default to 1 if dump_ep_context_model is true, otherwise 0
  auto embed_mode = std::stoi(session_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "-1"));
  if (embed_mode == -1) {
    if (info.dump_ep_context_model)
      embed_mode = 1;
    else
      embed_mode = 0;
  }

  if (0 <= embed_mode || embed_mode < 2) {
    info.ep_context_embed_mode = embed_mode;
  } else {
    ORT_THROW("Invalid ", kOrtSessionOptionEpContextEmbedMode, " must 0 or 1");
  }

  return info;
}

ProviderOptions NvExecutionProviderInfo::ToProviderOptions(const NvExecutionProviderInfo& info) {
  const ProviderOptions options{
      {nv::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {nv::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
      {nv::provider_option_names::kUserComputeStream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.user_compute_stream))},
      {nv::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {nv::provider_option_names::kMaxSharedMemSize, MakeStringWithClassicLocale(info.max_shared_mem_size)},
      {nv::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.dump_subgraphs)},
      {nv::provider_option_names::kEngineCacheEnable, MakeStringWithClassicLocale(info.engine_cache_enable)},
      {nv::provider_option_names::kEngineCachePath, MakeStringWithClassicLocale(info.engine_cache_path)},
      {nv::provider_option_names::kEngineCachePrefix, MakeStringWithClassicLocale(info.engine_cache_prefix)},
      {nv::provider_option_names::kEngineDecryptionEnable, MakeStringWithClassicLocale(info.engine_decryption_enable)},
      {nv::provider_option_names::kEngineDecryptionLibPath, MakeStringWithClassicLocale(info.engine_decryption_lib_path)},
      {nv::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.force_sequential_engine_build)},
      {nv::provider_option_names::kTimingCacheEnable, MakeStringWithClassicLocale(info.timing_cache_enable)},
      {nv::provider_option_names::kTimingCachePath, MakeStringWithClassicLocale(info.timing_cache_path)},
      {nv::provider_option_names::kForceTimingCacheMatch, MakeStringWithClassicLocale(info.force_timing_cache)},
      {nv::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.detailed_build_log)},
      {nv::provider_option_names::kProfilesMinShapes, MakeStringWithClassicLocale(info.profile_min_shapes)},
      {nv::provider_option_names::kProfilesMaxShapes, MakeStringWithClassicLocale(info.profile_max_shapes)},
      {nv::provider_option_names::kProfilesOptShapes, MakeStringWithClassicLocale(info.profile_opt_shapes)},
      {nv::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.cuda_graph_enable)},
      {nv::provider_option_names::kBuilderOptimizationLevel, MakeStringWithClassicLocale(info.builder_optimization_level)},
      {nv::provider_option_names::kUseExternalDataInitializer, MakeStringWithClassicLocale(info.use_external_data_initializer)},
      {nv::provider_option_names::kMultiProfileEnable, MakeStringWithClassicLocale(info.multi_profile_enable)},
      {nv::provider_option_names::kWeightStrippedEngineEnable, MakeStringWithClassicLocale(info.weight_stripped_engine_enable)},
      {nv::provider_option_names::kOnnxModelFolderPath, MakeStringWithClassicLocale(info.onnx_model_folder_path)},
      {nv::provider_option_names::kONNXBytestream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.onnx_bytestream))},
      {nv::provider_option_names::kONNXBytestreamSize, MakeStringWithClassicLocale(info.onnx_bytestream_size)},
      {nv::provider_option_names::kExternalDataBytestream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_data_bytestream))},
      {nv::provider_option_names::kExternalDataBytestreamSize, MakeStringWithClassicLocale(info.external_data_bytestream_size)}};
  return options;
}
}  // namespace onnxruntime
