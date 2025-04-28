// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_info.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
NvExecutionProviderInfo NvExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  NvExecutionProviderInfo info{};
  void* user_compute_stream = nullptr;
  void* onnx_bytestream = nullptr;
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
          .AddAssignmentToReference(nv::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
          .AddAssignmentToReference(nv::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
          .AddValueParser(
              nv::provider_option_names::kONNXBytestream,
              [&onnx_bytestream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                onnx_bytestream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kONNXBytestreamSize, info.onnx_bytestream_size)
          .Parse(options));  // add new provider option here.

  info.user_compute_stream = user_compute_stream;
  info.has_user_compute_stream = (user_compute_stream != nullptr);
  info.onnx_bytestream = onnx_bytestream;
  return info;
}

ProviderOptions NvExecutionProviderInfo::ToProviderOptions(const NvExecutionProviderInfo& info) {
  const ProviderOptions options{
      {nv::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {nv::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
      {nv::provider_option_names::kUserComputeStream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.user_compute_stream))},
      {nv::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {nv::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.dump_subgraphs)},
      {nv::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.detailed_build_log)},
      {nv::provider_option_names::kProfilesMinShapes, MakeStringWithClassicLocale(info.profile_min_shapes)},
      {nv::provider_option_names::kProfilesMaxShapes, MakeStringWithClassicLocale(info.profile_max_shapes)},
      {nv::provider_option_names::kProfilesOptShapes, MakeStringWithClassicLocale(info.profile_opt_shapes)},
      {nv::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.cuda_graph_enable)},
      {nv::provider_option_names::kONNXBytestream, MakeStringWithClassicLocale(info.onnx_bytestream)},
      {nv::provider_option_names::kONNXBytestreamSize, MakeStringWithClassicLocale(info.onnx_bytestream_size)},
  };
  return options;
}
}  // namespace onnxruntime
