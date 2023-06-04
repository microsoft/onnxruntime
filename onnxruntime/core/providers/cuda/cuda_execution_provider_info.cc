// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_provider_options.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "gpu_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kCudnnConvAlgoSearch = "cudnn_conv_algo_search";
constexpr const char* kDoCopyInDefaultStream = "do_copy_in_default_stream";
constexpr const char* kGpuExternalAlloc = "gpu_external_alloc";
constexpr const char* kGpuExternalFree = "gpu_external_free";
constexpr const char* kGpuExternalEmptyCache = "gpu_external_empty_cache";
constexpr const char* kCudnnConvUseMaxWorkspace = "cudnn_conv_use_max_workspace";
constexpr const char* kEnableCudaGraph = "enable_cuda_graph";
constexpr const char* kCudnnConv1dPadToNc1d = "cudnn_conv1d_pad_to_nc1d";
constexpr const char* kTunableOpEnable = "tunable_op_enable";
constexpr const char* kTunableOpTuningEnable = "tunable_op_tuning_enable";
constexpr const char* kTunableOpMaxTuningDurationMs = "tunable_op_max_tuning_duration_ms";
constexpr const char* kEnableSkipLayerNormStrictMode = "enable_skip_layer_norm_strict_mode";
}  // namespace provider_option_names
}  // namespace cuda

const EnumNameMapping<OrtCudnnConvAlgoSearch> ort_cudnn_conv_algo_search_mapping{
    {OrtCudnnConvAlgoSearchExhaustive, "EXHAUSTIVE"},
    {OrtCudnnConvAlgoSearchHeuristic, "HEURISTIC"},
    {OrtCudnnConvAlgoSearchDefault, "DEFAULT"},
};

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};

CUDAExecutionProviderInfo CUDAExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  CUDAExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;
  void* empty_cache = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              cuda::provider_option_names::kDeviceId,
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
          .AddValueParser(
              cuda::provider_option_names::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              cuda::provider_option_names::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              cuda::provider_option_names::kGpuExternalEmptyCache,
              [&empty_cache](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(cuda::provider_option_names::kMemLimit, info.gpu_mem_limit)
          .AddAssignmentToEnumReference(
              cuda::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .AddAssignmentToEnumReference(
              cuda::provider_option_names::kCudnnConvAlgoSearch,
              ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)
          .AddAssignmentToReference(cuda::provider_option_names::kDoCopyInDefaultStream, info.do_copy_in_default_stream)
          .AddAssignmentToReference(cuda::provider_option_names::kCudnnConvUseMaxWorkspace, info.cudnn_conv_use_max_workspace)
          .AddAssignmentToReference(cuda::provider_option_names::kEnableCudaGraph, info.enable_cuda_graph)
          .AddAssignmentToReference(cuda::provider_option_names::kCudnnConv1dPadToNc1d, info.cudnn_conv1d_pad_to_nc1d)
          .AddAssignmentToReference(cuda::provider_option_names::kEnableSkipLayerNormStrictMode, info.enable_skip_layer_norm_strict_mode)
          .AddValueParser(
              cuda::provider_option_names::kTunableOpEnable,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.enable));
                return Status::OK();
              })
          .AddValueParser(
              cuda::provider_option_names::kTunableOpTuningEnable,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.tuning_enable));
                return Status::OK();
              })
          .AddValueParser(
              cuda::provider_option_names::kTunableOpMaxTuningDurationMs,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.tunable_op.max_tuning_duration_ms));
                return Status::OK();
              })
          .Parse(options));

  CUDAExecutionProviderExternalAllocatorInfo alloc_info{alloc, free, empty_cache};
  info.external_allocator_info = alloc_info;
  return info;
}

ProviderOptions CUDAExecutionProviderInfo::ToProviderOptions(const CUDAExecutionProviderInfo& info) {
  const ProviderOptions options{
      {cuda::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cuda::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
      {cuda::provider_option_names::kGpuExternalAlloc, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.alloc))},
      {cuda::provider_option_names::kGpuExternalFree, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.free))},
      {cuda::provider_option_names::kGpuExternalEmptyCache, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache))},
      {cuda::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {cuda::provider_option_names::kCudnnConvAlgoSearch,
       EnumToName(ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)},
      {cuda::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {cuda::provider_option_names::kCudnnConvUseMaxWorkspace, MakeStringWithClassicLocale(info.cudnn_conv_use_max_workspace)},
      {cuda::provider_option_names::kEnableCudaGraph, MakeStringWithClassicLocale(info.enable_cuda_graph)},
      {cuda::provider_option_names::kCudnnConv1dPadToNc1d, MakeStringWithClassicLocale(info.cudnn_conv1d_pad_to_nc1d)},
      {cuda::provider_option_names::kTunableOpEnable, MakeStringWithClassicLocale(info.tunable_op.enable)},
      {cuda::provider_option_names::kTunableOpTuningEnable, MakeStringWithClassicLocale(info.tunable_op.tuning_enable)},
      {cuda::provider_option_names::kTunableOpMaxTuningDurationMs, MakeStringWithClassicLocale(info.tunable_op.max_tuning_duration_ms)},
      {cuda::provider_option_names::kEnableSkipLayerNormStrictMode, MakeStringWithClassicLocale(info.enable_skip_layer_norm_strict_mode)},
  };

  return options;
}

ProviderOptions CUDAExecutionProviderInfo::ToProviderOptions(const OrtCUDAProviderOptionsV2& info) {
  const ProviderOptions options{
      {cuda::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {cuda::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
      {cuda::provider_option_names::kArenaExtendStrategy, EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
      {cuda::provider_option_names::kCudnnConvAlgoSearch, EnumToName(ort_cudnn_conv_algo_search_mapping, info.cudnn_conv_algo_search)},
      {cuda::provider_option_names::kDoCopyInDefaultStream, MakeStringWithClassicLocale(info.do_copy_in_default_stream)},
      {cuda::provider_option_names::kCudnnConvUseMaxWorkspace, MakeStringWithClassicLocale(info.cudnn_conv_use_max_workspace)},
      {cuda::provider_option_names::kCudnnConv1dPadToNc1d, MakeStringWithClassicLocale(info.cudnn_conv1d_pad_to_nc1d)},
      {cuda::provider_option_names::kTunableOpEnable, MakeStringWithClassicLocale(info.tunable_op_enable)},
      {cuda::provider_option_names::kTunableOpTuningEnable, MakeStringWithClassicLocale(info.tunable_op_tuning_enable)},
      {cuda::provider_option_names::kTunableOpMaxTuningDurationMs, MakeStringWithClassicLocale(info.tunable_op_max_tuning_duration_ms)},
  };

  return options;
}

}  // namespace onnxruntime
