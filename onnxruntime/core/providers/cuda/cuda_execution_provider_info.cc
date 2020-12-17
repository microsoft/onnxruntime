// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_execution_provider_info.h"

#include "core/common/string_utils.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "cuda_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kCudnnConvAlgo = "cudnn_conv_algo";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
}  // namespace provider_option_names

CUDAExecutionProviderInfo CUDAExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  CUDAExecutionProviderInfo info{};

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseString(value_str, info.cuda_mem_limit));
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
          .AddAssignmentToReference(provider_option_names::kMemLimit, info.cuda_mem_limit)
          .AddAssignmentToReference(provider_option_names::kArenaExtendStrategy, info.arena_extend_strategy)
          .AddValueParser(
              provider_option_names::kCudnnConvAlgo,
              [&info](const std::string& value_str) -> Status {
                int cudnn_conv_algo_val{};
                ORT_RETURN_IF_ERROR(ParseString(value_str, cudnn_conv_algo_val));
                ORT_RETURN_IF_NOT(
                    cudnn_conv_algo_val == OrtCudnnConvAlgoSearch::EXHAUSTIVE ||
                        cudnn_conv_algo_val == OrtCudnnConvAlgoSearch::HEURISTIC ||
                        cudnn_conv_algo_val == OrtCudnnConvAlgoSearch::DEFAULT,
                    "Invalid OrtCudnnConvAlgoSearch value: ", cudnn_conv_algo_val);
                info.cudnn_conv_algo = static_cast<OrtCudnnConvAlgoSearch>(cudnn_conv_algo_val);
                return Status::OK();
              })
          .AddAssignmentToReference(provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .Parse(options));

  return info;
}

ProviderOptions CUDAExecutionProviderInfo::ToProviderOptions(const CUDAExecutionProviderInfo& info) {
  const ProviderOptions options{
      {provider_option_names::kDeviceId, MakeString(info.device_id)},
      {provider_option_names::kMemLimit, MakeString(info.cuda_mem_limit)},
      {provider_option_names::kArenaExtendStrategy, MakeString(info.arena_extend_strategy)},
      {provider_option_names::kCudnnConvAlgo, MakeString(static_cast<int>(info.cudnn_conv_algo))},
      {provider_option_names::kDoCopyInDefaultStream, MakeString(info.do_copy_in_default_stream)},
  };

  return options;
}
}  // namespace onnxruntime
