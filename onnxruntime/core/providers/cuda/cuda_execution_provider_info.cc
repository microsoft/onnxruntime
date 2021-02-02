// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "cuda_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kCudnnConvAlgoSearch = "cudnn_conv_algo_search";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
}  // namespace provider_option_names
}  // namespace cuda

namespace {
const EnumNameMapping<OrtCudnnConvAlgoSearch> ort_cudnn_conv_algo_search_mapping{
    {OrtCudnnConvAlgoSearch::EXHAUSTIVE, "EXHAUSTIVE"},
    {OrtCudnnConvAlgoSearch::HEURISTIC, "HEURISTIC"},
    {OrtCudnnConvAlgoSearch::DEFAULT, "DEFAULT"},
};

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};
}  // namespace

CUDAExecutionProviderInfo CUDAExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  CUDAExecutionProviderInfo info{};

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              cuda::provider_option_names::kDeviceId,
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
          .AddAssignmentToReference(cuda::provider_option_names::kMemLimit, info.cuda_mem_limit)
          .AddAssignmentToEnumReference(
              cuda::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .AddAssignmentToEnumReference(
              cuda::provider_option_names::kCudnnConvAlgoSearch,
              ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)
          .AddAssignmentToReference(cuda::provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .Parse(options));

  return info;
}

ProviderOptions CUDAExecutionProviderInfo::ToProviderOptions(const CUDAExecutionProviderInfo& info) {
  const ProviderOptions options{
      {cuda::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cuda::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.cuda_mem_limit)},
      {cuda::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {cuda::provider_option_names::kCudnnConvAlgoSearch,
       EnumToName(ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)},
      {cuda::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
  };

  return options;
}
}  // namespace onnxruntime
