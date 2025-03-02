// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/common.h"
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
  /**
   * Get the GraphOptimizerRegistry instance.
   * Note: Please call Create() to create the GraphOptimizerRegistry instance before using Get().
   */
  static const GraphOptimizerRegistry& Get() {
    ORT_ENFORCE(graph_optimizer_registry_);
    return *graph_optimizer_registry_;
  }

  /**
   * Create and initialize the graph optimizer registry instance as a singleton.
   * The registry also keeps the references to session options, cpu_ep and logger tha are required by some optimizers.
   */
  static Status Create(const onnxruntime::SessionOptions* sess_options,
                       const onnxruntime::IExecutionProvider* cpu_ep,
                       const logging::Logger* logger);

  /**
   * Get optimizer selection function. If the optimizer name can't be found, return nullopt.
   */
  std::optional<SelectionFunc> GetSelectionFunc(std::string& name) const;

  /**
   * Get CPU EP.
   */
  const onnxruntime::IExecutionProvider& GetCpuEp() const { return *cpu_ep_; }

  /**
   * Get Session Options.
   */
  const onnxruntime::SessionOptions& GetSessionOptions() const { return *session_options_; }

  /**
   * Get Logger.
   */
  const logging::Logger* GetLogger() const { return logger_; }

 private:
  const onnxruntime::SessionOptions* session_options_;
  const onnxruntime::IExecutionProvider* cpu_ep_;
  const logging::Logger* logger_;

  static std::unique_ptr<GraphOptimizerRegistry> graph_optimizer_registry_;
  static std::mutex registry_mutex_;

  InlinedHashMap<std::string, SelectionFunc> transformer_name_to_selection_func_;

  GraphOptimizerRegistry(const onnxruntime::SessionOptions* sess_options,
                         const onnxruntime::IExecutionProvider* cpu_ep,
                         const logging::Logger* logger);

  /**
   * Create pre-defined selection functions.
   */
  Status CreatePredefinedSelectionFuncs();
};
}  // namespace onnxruntime
