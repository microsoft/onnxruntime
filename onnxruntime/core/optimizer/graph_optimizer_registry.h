// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
using KeyValueConfig = std::unordered_map<std::string, std::string>;
using SelectionFunc = std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&, const KeyValueConfig&)>;

/**
 * A registration/lookup class for re-usable optimizers for EPs.
 */
class GraphOptimizerRegistry {
 public:
  explicit GraphOptimizerRegistry();
  GraphOptimizerRegistry(const GraphOptimizerRegistry&) = delete;

  /**
   * Get GraphOptimizerRegistry instance as a singleton.
   */
  static std::shared_ptr<GraphOptimizerRegistry> Get() {
    if (!graph_optimizer_registry) {  // First Check (without locking)
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (!graph_optimizer_registry) {  // Second Check (with locking)
        graph_optimizer_registry = std::make_shared<GraphOptimizerRegistry>();
      }
    }
    return graph_optimizer_registry;
  }

  /**
   * Initialize the graph optimizer registry as well as add predefined optimizers and selection functions for later lookup.
   * The registry also keeps the references to session options, cpu_ep and logger tha are required by some optimizers.
   */
  common::Status Create(const onnxruntime::SessionOptions* sess_options,
                        const onnxruntime::IExecutionProvider* cpu_ep,
                        const logging::Logger* logger);

  /**
   * Get optimizer selection function. If the optimizer name can't be found, return nullopt.
   */
  std::optional<SelectionFunc> GetSelectionFunc(std::string& name) const;

  /**
   * Get CPU EP reference.
   */
  const onnxruntime::IExecutionProvider* GetCpuEpReference() const { return cpu_ep_; }

  /**
   * Get Session Options reference.
   */
  const onnxruntime::SessionOptions* GetSessionOptionsReference() const { return session_options_; }

 private:
  InlinedHashMap<std::string, SelectionFunc> transformer_name_to_selection_func_;
  const logging::Logger* logger_;
  const onnxruntime::IExecutionProvider* cpu_ep_;
  const onnxruntime::SessionOptions* session_options_;

  static std::shared_ptr<GraphOptimizerRegistry> graph_optimizer_registry;
  static std::mutex registry_mutex;
};
}  // namespace onnxruntime
