// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_info.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "nv_provider_options_internal.h"

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

ProviderOptions NvExecutionProviderInfo::ToProviderOptions(const OrtNvTensorRtRtxProviderOptions& info) {
  auto empty_if_null = [](const char* s) { return s != nullptr ? std::string{s} : std::string{}; };

  const std::string kProfilesMinShapes_ = empty_if_null(info.profile_min_shapes);
  const std::string kProfilesMaxShapes_ = empty_if_null(info.profile_max_shapes);
  const std::string kProfilesOptShapes_ = empty_if_null(info.profile_opt_shapes);

  const ProviderOptions options{
      {nv::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {nv::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
      {nv::provider_option_names::kUserComputeStream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.user_compute_stream))},
      {nv::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {nv::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.dump_subgraphs)},
      {nv::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.detailed_build_log)},
      {nv::provider_option_names::kProfilesMinShapes, kProfilesMinShapes_},
      {nv::provider_option_names::kProfilesMaxShapes, kProfilesMaxShapes_},
      {nv::provider_option_names::kProfilesOptShapes, kProfilesOptShapes_},
      {nv::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.cuda_graph_enable)},
      {nv::provider_option_names::kONNXBytestream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.onnx_bytestream))},
      {nv::provider_option_names::kONNXBytestreamSize, MakeStringWithClassicLocale(info.onnx_bytestream_size)},

  };
  return options;
}

/**
 * Update OrtNvTensorRtRtxProviderOptions instance with ProviderOptions (map of string-based key-value pairs)
 *
 * Please note that it will reset the OrtNvTensorRtRtxProviderOptions instance first and then set up the provided provider options
 * See NvExecutionProviderInfo::FromProviderOptions() for more details. This function will be called by the C API UpdateTensorRTProviderOptions() also.
 *
 * \param provider_options - a pointer to OrtNvTensorRtRtxProviderOptions instance
 * \param options - a reference to ProviderOptions instance
 * \param string_copy - if it's true, it uses strncpy() to copy 'provider option' string from ProviderOptions instance to where the 'provider option' const char pointer in OrtNvTensorRtRtxProviderOptions instance points to.
 *                      it it's false, it only saves the pointer and no strncpy().
 *
 * Note: If there is strncpy involved, please remember to deallocate or simply call C API ReleaseTensorRTProviderOptions.
 */
void NvExecutionProviderInfo::UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) {
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

  NvExecutionProviderInfo internal_options = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(options);
  auto& nv_provider_options = *reinterpret_cast<OrtNvTensorRtRtxProviderOptions*>(provider_options);
  nv_provider_options.device_id = internal_options.device_id;

  // The 'has_user_compute_stream' of the OrtNvTensorRtRtxProviderOptions instance can be set by C API UpdateTensorRTProviderOptionsWithValue() as well
  // We only set the 'has_user_compute_stream' of the OrtNvTensorRtRtxProviderOptions instance if it is provided in options or user_compute_stream is provided
  if (options.find("has_user_compute_stream") != options.end()) {
    nv_provider_options.has_user_compute_stream = internal_options.has_user_compute_stream;
  }
  if (options.find("user_compute_stream") != options.end() && internal_options.user_compute_stream != nullptr) {
    nv_provider_options.user_compute_stream = internal_options.user_compute_stream;
    nv_provider_options.has_user_compute_stream = true;
  }

  nv_provider_options.max_workspace_size = internal_options.max_workspace_size;
  nv_provider_options.dump_subgraphs = internal_options.dump_subgraphs;
  nv_provider_options.detailed_build_log = internal_options.detailed_build_log;
  nv_provider_options.profile_min_shapes = copy_string_if_needed(internal_options.profile_min_shapes);
  nv_provider_options.profile_max_shapes = copy_string_if_needed(internal_options.profile_max_shapes);
  nv_provider_options.profile_opt_shapes = copy_string_if_needed(internal_options.profile_opt_shapes);

  nv_provider_options.cuda_graph_enable = internal_options.cuda_graph_enable;
  nv_provider_options.onnx_bytestream = internal_options.onnx_bytestream;
  nv_provider_options.onnx_bytestream_size = internal_options.onnx_bytestream_size;
}
}  // namespace onnxruntime
