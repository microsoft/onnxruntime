// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "onnxruntime_c_api.h"
#include "core/framework/arena_extend_strategy.h"

struct OrtCANNProviderOptions {
  int device_id{0};                                           // CANN device id
  size_t npu_mem_limit{SIZE_MAX};                             // BFC Arena memory limit for CANN
  onnxruntime::ArenaExtendStrategy arena_extend_strategy{
      static_cast<onnxruntime::ArenaExtendStrategy>(0)};      // Strategy used to grow the memory arena
  int enable_cann_graph{1};                                   // Flag indicating if prioritizing the use of
                                                              // CANN's graph-running capabilities
  int enable_cann_subgraph{0};                                // Flag indicating whether to generate subgraph
                                                              // automaticly
  int dump_graphs{0};                                         // Flag indicating if dumping graphs
  int dump_om_model{1};                                       // Flag indicating if dumping om model
  std::string precision_mode;                                 // Operator Precision Mode
  std::string op_select_impl_mode;                            // Operator-level model compilation options:
                                                              // Mode selection
  std::string optypelist_for_implmode;                        // Operator-level model compilation options:
                                                              // Operator list
  OrtArenaCfg* default_memory_arena_cfg{nullptr};             // CANN memory arena configuration parameters
};
