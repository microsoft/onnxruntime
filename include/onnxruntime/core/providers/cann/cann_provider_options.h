// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "onnxruntime_c_api.h"
#include "core/framework/arena_extend_strategy.h"

struct OrtCANNProviderOptions {
  int device_id;                                           // CANN device id
  size_t npu_mem_limit;                                    // BFC Arena memory limit for CANN
  onnxruntime::ArenaExtendStrategy arena_extend_strategy;  // Strategy used to grow the memory arena
  int enable_cann_graph;                                   // Flag indicating if prioritizing the use of
                                                           // CANN's graph-running capabilities
  int enable_cann_subgraph;                                // Flag indicating whether to generate subgraph
                                                           // automaticly
  int dump_graphs;                                         // Flag indicating if dumping graphs
  int dump_om_model;                                       // Flag indicating if dumping om model
  std::string precision_mode_v2;                           // Operator Precision Mode
  std::string op_select_impl_mode;                         // Operator-level model compilation options:
                                                           // Mode selection
  std::string optypelist_for_implmode;                     // Operator-level model compilation options:
                                                           // Operator list
  std::string input_format;                                // Input format, e.g., "NCHW"
  std::string dynamic_batch_size;                          // Dynamic batch size, e.g., "2,4,8"
  std::string dynamic_image_size;                          // Dynamic image size, e.g., "224,224;256,256"
  std::string dynamic_dims;                                // Dynamic dims, e.g., "1,3,224,224;1,3,256,256"
  OrtArenaCfg* default_memory_arena_cfg;                   // CANN memory arena configuration parameters
};
