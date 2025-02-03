// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
//#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"

namespace onnxruntime {
class GraphOptimizerRegistry {
 public:
  explicit GraphOptimizerRegistry() {}
  GraphOptimizerRegistry(const GraphOptimizerRegistry&) = delete;

  static std::shared_ptr<GraphOptimizerRegistry> Get() {
    if (!graph_optimizer_registry) { // First Check (without locking)
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (!graph_optimizer_registry) { // Second Check (with locking)
        graph_optimizer_registry = std::make_shared<GraphOptimizerRegistry>();
      }
    }
    return graph_optimizer_registry;
  }

  common::Status AddPredefinedOptimizers(const onnxruntime::SessionOptions& sess_options,
                        const onnxruntime::IExecutionProvider& cpu_ep,
                        const logging::Logger& logger);

  common::Status ApplyTransformer(Graph& graph, std::string& name,
                                  const logging::Logger& logger) const;

  common::Status Register(std::unique_ptr<GraphTransformer> transformer);
  
  // Get transformer by name. Return nullptr if not found.
  GraphTransformer* GetTransformerByName(std::string& name) const;

 private:
  InlinedVector<std::unique_ptr<GraphTransformer>> transformer_list_;
  InlinedHashMap<std::string, GraphTransformer*> name_to_transformer_map_;

  static std::shared_ptr<GraphOptimizerRegistry> graph_optimizer_registry;
  static std::mutex registry_mutex;
};
}  // namespace onnxruntime
