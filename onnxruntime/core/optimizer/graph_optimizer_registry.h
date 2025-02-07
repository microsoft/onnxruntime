// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
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
    if (!graph_optimizer_registry) { // First Check (without locking)
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (!graph_optimizer_registry) { // Second Check (with locking)
        graph_optimizer_registry = std::make_shared<GraphOptimizerRegistry>();
      }
    }
    return graph_optimizer_registry;
  }

  /**
   * Register all the predefined optimizer names, only name not the optimizer instance.
   * 
   * The optimizer will later be instantizted only when EP requests it by calling GetOptimizerByName in provider bridge.
   */
  common::Status GraphOptimizerRegistry::AddPredefinedOptimizerNames(std::vector<std::string>& optimizer_names);

  /**
   * Create and register all predefined optimizers.
   */
  common::Status AddPredefinedOptimizers(const onnxruntime::SessionOptions& sess_options,
                                         const onnxruntime::IExecutionProvider& cpu_ep,
                                         const logging::Logger& logger);

  /**
   * Create and register optimizer. 
   */
  common::Status GraphOptimizerRegistry::CreateOptimizer(std::string& name, std::unordered_map<std::string, std::string>& key_value_configs);

  /**
   * Get optimizer by name.
   */
  GraphTransformer* GraphOptimizerRegistry::GetTransformerByName(std::string& name) const;

  /**
   * Run the optimizer. 
   */
  common::Status ApplyTransformer(Graph& graph, std::string& name,
                                  const logging::Logger& logger) const;

  /**
   * Register optimizer and its optimization selection function.
   */
  common::Status Register(std::unique_ptr<GraphTransformer> transformer);

  /**
   * Get optimizer selection function. If the optimizer name can't be found, return nullopt.
   */
  std::optional<std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&)>> GraphOptimizerRegistry::GetSelectionFunc(std::string& name) const;

  /**
   * Add CPU EP reference from InferenceSession as it's needed for some optimizers, ex: ConstantFoldingDQ.
   */
  common::Status AddCpuEpReference(onnxruntime::IExecutionProvider* cpu_ep);

  /**
   * Get CPU EP reference.
   */
  onnxruntime::IExecutionProvider* GetCpuEpReference() const { return cpu_ep_; }

  /**
   * Add session options reference from InferenceSession as it's needed for some optimizers, ex: ConstantFoldingDQ.
   */
  common::Status AddSessionOptionsReference(onnxruntime::SessionOptions* session_options);

  /**
   * Get Session Options reference.
   */
  onnxruntime::SessionOptions* GetSessionOptionsReference() const { return session_options_; }

 private:
  InlinedVector<std::unique_ptr<GraphTransformer>> transformer_list_;
  InlinedHashMap<std::string, GraphTransformer*> name_to_transformer_map_;
  InlinedHashMap<std::string, std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&)>> transformer_name_to_selection_func_;
  const logging::Logger* logger_;
  onnxruntime::IExecutionProvider* cpu_ep_;
  onnxruntime::SessionOptions* session_options_;

  static std::shared_ptr<GraphOptimizerRegistry> graph_optimizer_registry;
  static std::mutex registry_mutex;
};
}  // namespace onnxruntime
