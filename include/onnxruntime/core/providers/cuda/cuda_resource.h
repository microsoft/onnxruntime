// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/resource.h"

#define ORT_CUDA_RESOUCE_VERSION 3

enum CudaResource : int {
  cuda_stream_t = cuda_resource_offset,  // 10000
  cudnn_handle_t,
  cublas_handle_t,
  deferred_cpu_allocator_t,
  // below are cuda ep options
  device_id_t,  // 10004
  has_user_compute_stream_t,
  gpu_mem_limit_t,
  arena_extend_strategy_t,
  cudnn_conv_algo_search_t,  // 10008
  do_copy_in_default_stream_t,
  gpu_external_alloc_t,
  gpu_external_free_t,
  gpu_external_empty_cache_t,  // 10012
  cudnn_conv_use_max_workspace_t,
  enable_cuda_graph_t,
  cudnn_conv1d_pad_to_nc1d_t,
  tunable_op_enable_t,  // 10016
  tunable_op_tuning_enable_t,
  tunable_op_max_tuning_duration_ms_t,
  enable_skip_layer_norm_strict_mode_t,
  prefer_nhwc_t,  // 10020
  use_ep_level_unified_stream_t
};