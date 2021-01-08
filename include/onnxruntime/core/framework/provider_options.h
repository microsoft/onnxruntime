// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "core/session/onnxruntime_c_api.h"

/// <summary>
/// This is CUDA provider specific but needs to live in a header that is build-flavor agnostic
/// Options for the CUDA provider that are passed to SessionOptionsAppendExecutionProvider_CUDA
/// </summary>
typedef struct OrtCUDAProviderOptions {
  int device_id;                                  // cuda device with id=0 as default device.
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search;  // cudnn conv algo search option
  size_t cuda_mem_limit;                          // default cuda memory limitation to maximum finite value of size_t.
  int arena_extend_strategy;                      // default area extend strategy to KNextPowerOfTwo.
  int do_copy_in_default_stream;                  // default true
} OrtCUDAProviderOptions;

namespace onnxruntime {

// data types for execution provider options

using ProviderOptions = std::unordered_map<std::string, std::string>;
using ProviderOptionsVector = std::vector<ProviderOptions>;
using ProviderOptionsMap = std::unordered_map<std::string, ProviderOptions>;

}  // namespace onnxruntime
