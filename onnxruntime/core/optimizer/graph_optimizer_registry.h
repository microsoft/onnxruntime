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
/**
 * Optimizer's selection function: Selects a set of nodes from a given graph for optimization. Additional key/value strings can be provided to configure the optimizer.
 *                                 If needed, use graph_optimizer_registry to access the session options, the CPU EP and the logger.
 *
 * Optimizer's optimization function: Gets the nodes in ComputeCapability from nodes_to_optimize. Use graph_optimizer_registry to access the session options, the CPU EP
 *                                    and the logger if needed to create the optimizer. Run optimization on the nodes/subgraph, and finally, update the ComputeCapability.
 *
 */
using KeyValueConfig = std::unordered_map<std::string, std::string>;
using SelectionFunc = std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&,
                                                                                    const KeyValueConfig&,
                                                                                    const GraphOptimizerRegistry& graph_optimizer_registry)>;
using OptimizationFunc = std::function<Status(Graph& graph,
                                              const ComputeCapability& optimization_cc,
                                              ComputeCapability& cc_to_update,
                                              const GraphOptimizerRegistry& graph_optimizer_registry)>;

/**
 * A registration/lookup class for re-usable optimizers for EPs.
 */
class GraphOptimizerRegistry {
 public:
  /**
   * The constructor takes in session options, the CPU EP and a logger as these are required by some optimizers.
   */
  GraphOptimizerRegistry(const onnxruntime::SessionOptions* sess_options,
                         const onnxruntime::IExecutionProvider* cpu_ep,
                         const logging::Logger* logger,
                         bool capability_optimizations_supported = true);

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

  // Describes the capability consumer, not a model filename or configuration hint.
  // Consumers that do not execute capability optimization callbacks must opt out.
  bool SupportsCapabilityOptimizations() const noexcept { return capability_optimizations_supported_; }

 private:
  const onnxruntime::SessionOptions* session_options_;
  const onnxruntime::IExecutionProvider* cpu_ep_;
  const logging::Logger* logger_;
  const bool capability_optimizations_supported_;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  InlinedHashMap<std::string, SelectionFunc> transformer_name_to_selection_func_;

  /**
   * Create pre-defined selection functions.
   */
  Status CreatePredefinedSelectionFuncs();
#endif
};
}  // namespace onnxruntime
