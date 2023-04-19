// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "core/framework/arena_extend_strategy.h"

/// <summary>
/// Options for the CUDA provider that are passed to SessionOptionsAppendExecutionProvider_CUDA_V2.
/// Please note that this struct is *similar* to OrtCUDAProviderOptions but only to be used internally.
/// Going forward, new cuda provider options are to be supported via this struct and usage of the publicly defined
/// OrtCUDAProviderOptions will be deprecated over time.
/// User can only get the instance of OrtCUDAProviderOptionsV2 via CreateCUDAProviderOptions.
/// </summary>
struct OrtCUDAProviderOptionsV2 {
  int device_id;                                           // cuda device id.
  int has_user_compute_stream;                             // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                               // user specified CUDA compute stream.
  int do_copy_in_default_stream;                           // flag specifying if the default stream is to be used for copying.
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search;           // cudnn algo search enum.
  size_t gpu_mem_limit;                                    // BFC Arena memory limit for CUDA.
                                                           // (will be overridden by contents of `default_memory_arena_cfg` is it exists)
  onnxruntime::ArenaExtendStrategy arena_extend_strategy;  // BFC Arena extension strategy.
                                                           // (will be overridden by contents of `default_memory_arena_cfg` is it exists)
  OrtArenaCfg* default_memory_arena_cfg;                   // BFC Arena config flags.
  int cudnn_conv_use_max_workspace;                        // flag specifying if maximum workspace can be used in cudnn conv algo search.
  int enable_cuda_graph;                                   // flag specifying if the CUDA graph is to be captured for the model.
  int cudnn_conv1d_pad_to_nc1d;                            // flag specifying if pad Conv1D's input [N,C,D] to [N,C,1,D] or [N,C,D,1].
  int tunable_op_enabled;                                  // flag specifying if TunableOp is enabled.
};
