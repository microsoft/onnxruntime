// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

class GraphOptimizer {
 public:
  GraphOptimizer(Graph& graph, logging::Logger& logger) : graph_(graph),
                                                          logger_(logger) {
  }

  Status Init();
  AllocatorPtr GetAllocator() {
    return allocator_ptr_;
  }
  const MLValue* GetNodeInputOrOutputMLValue(const std::string& mlvalue_name) const;
  MLValue* GetMutableNodeInputOrOutputMLValue(const std::string& mlvalue_name);
  Status GetOrCreateNodeOutputMLValue(const NodeArg* node_arg,
                                      const MLValueAllocationParameters& parameters,
                                      MLValue*& p_mlvalue);

private:
  common::Status InitMLValueNameIndexMapping();
  Status InitMLValues();

  Graph& graph_;
  logging::Logger& logger_;

  // The optimizer is running on CPU execution provider by default.
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider_;
  const int device_id_{0};
  const OrtMemType mem_type_{OrtMemTypeDefault};
  AllocatorPtr allocator_ptr_;

  // MLValues for optimizer
  MLValueNameIdxMap mlvalue_name_idx_map_;
  std::vector<MLValue> all_values_;
};

}  // namespace onnxruntime