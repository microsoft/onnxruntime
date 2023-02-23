// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "core/framework/arena_extend_strategy.h"

struct OrtCANNProviderOptions {
  int device_id;                                           // CANN device id
  size_t npu_mem_limit;                                    // BFC Arena memory limit for CANN
  onnxruntime::ArenaExtendStrategy arena_extend_strategy;  // Strategy used to grow the memory arena
  int do_copy_in_default_stream;                           // Flag indicating if copying needs to take place on the
                                                           // same stream as the compute stream in the CANN EP
  int enable_cann_graph;                                   // Flag indicating if prioritizing the use of
                                                           // CANN's graph-running capabilities
  OrtArenaCfg* default_memory_arena_cfg;                   // CANN memory arena configuration parameters
};
