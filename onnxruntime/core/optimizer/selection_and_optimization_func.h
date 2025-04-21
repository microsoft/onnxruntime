// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
static const std::string kConstantFoldingDQ = "ConstantFoldingDQ";

/**
 * Optimizer's selection function: Selects a set of nodes from a given graph for optimization. Additional key/value strings can be provided to configure the optimizer.
 *                                 If needed, use graph_optimizer_registry to access the session options, the CPU EP and the logger.
 *
 * Optimizer's optimization function: Gets the nodes in ComputeCapability from nodes_to_optimize. Use graph_optimizer_registry to access the session options, the CPU EP
 *                                    and the logger if needed to create the optimizer. Run optimization on the nodes/subgraph, and finally, update the ComputeCapability.
 *
 */

struct ConstantFoldingDQFuncs {
  static std::vector<std::unique_ptr<ComputeCapability>> Select(const GraphViewer& graph_viewer,
                                                                const KeyValueConfig& configs,
                                                                const GraphOptimizerRegistry& graph_optimizer_registry);
  static Status Optimize(Graph& graph,
                         const ComputeCapability& optimization_cc,
                         ComputeCapability& cc_to_update,
                         const GraphOptimizerRegistry& graph_optimizer_registry);
};
}  // namespace onnxruntime
