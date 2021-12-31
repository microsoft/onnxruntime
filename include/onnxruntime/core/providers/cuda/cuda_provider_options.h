// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

/// <summary>
/// Options for the CUDA provider that are passed to SessionOptionsAppendExecutionProvider_Cuda_V2.
/// Please note that this struct is *similar* to OrtCUDAProviderOptions but only to be used internally.
/// Going forward, new cuda provider options are to be supported via this struct and usage of the publicly defined
/// OrtCUDAProviderOptions will be deprecated over time.
/// User can only get the instance of OrtCUDAProviderOptionsV2 via CreateCUDAProviderOptions.
/// </summary>
struct OrtCUDAProviderOptionsV2 {
  int device_id;                                // cuda device id.
  int has_user_compute_stream;                  // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                    // user specified CUDA compute stream.
  int do_copy_in_default_stream;
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search;
  size_t gpu_mem_limit;
  int arena_extend_strategy;
  OrtArenaCfg* default_memory_arena_cfg;
  int cudnn_conv_use_max_workspace;
};
