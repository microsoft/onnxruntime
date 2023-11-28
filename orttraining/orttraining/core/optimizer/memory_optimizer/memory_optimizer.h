// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/optimizer/graph_transformer.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_insight.h"

namespace onnxruntime {

/**
@Class MemoryOptimizer

Find recompute subgraphs and enable them according to user configs. The way we collect subgraphs
(in orttraining/orttraining/core/optimizer/memory_optimizer/recompute_analysis.h) in brief is:
1. Find all nodes that generate stashed activations.
2. For each node, check it data type is supported to recompute
  a. If yes, add it in the subgraph, and append its input in the queue to scan next;
  b. otherwise, stop collecting and return the subgraph (could be empty).
3. Pick up the input node from the queue, and do 2 again. The process ends when the queue is empty or 2.b happens.
4. Clone the recomputable subgraphs with lower node priority (to execute) and insert them back to the original graph.
*/

class MemoryOptimizer : public GraphTransformer {
 private:
 public:
  MemoryOptimizer(const std::string& memory_optimizer_config, const std::string& recompute_probe_config)
      : GraphTransformer("MemoryOptimizer") {
    // Parse user-defined configs.
    ORT_ENFORCE(ParseConfigFromString(memory_optimizer_config, recompute_probe_config).IsOK());
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status ParseConfigFromString(const std::string& memory_optimizer_config, const std::string& recompute_probe_config);

  /**
   * @brief Apply graph modifications based on user configs.
   *
   * @param graph Graph to iterate and modify.
   * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
   *   Used to re-order the collected subgraph nodes.
   * @param candidate_output_args_map  A map from node to its candidate activations, which are consumed by both fw and
   *  bw ops.
   * @param logger Logger.
   * @param boundary_op_order_in_topological_sort index of the boundary op between fw and bw.
   * @param subgraph_stores  A store to maintain all found subgraphs.
   * @param node The node we used to look for corresponding optimization graphs.
   * @return true
   * @return false
   */
  bool ModifyGraph(Graph& graph,
                   const InlinedHashMap<NodeIndex, ptrdiff_t>&
                       node_index_to_its_order_in_topological_sort_map,
                   const InlinedHashMap<const Node*, InlinedVector<size_t>>&
                       candidate_output_args_map,
                   const logging::Logger& logger,
                   ptrdiff_t boundary_op_order_in_topological_sort,
                   Node* node,
                   std::shared_ptr<optimizer::memory_optimizer::NodeOptimizationPlanBase>& node_plan,
                   std::shared_ptr<optimizer::memory_optimizer::ClusterApplyContext>& apply_context) const;

  /**
   * @brief Summarize transformation details.
   *
   * @param stashed_activation_statistics statistics around stashed activation memory saving.
   * @return void
   */
  void PrintSummary(const optimizer::memory_optimizer::MemoryOptimizationPlanner& mem_opt_stats,
                    const InlinedHashMap<const Node*,
                                         std::shared_ptr<optimizer::memory_optimizer::ClusterApplyContext>>&
                        node_to_apply_contexts_map,
                    const logging::Logger& logger) const;

  /**************************************************
   ** Recompute related function definition starts **
   *************************************************/

  /**
   * @brief Duplicate nodes to create a recompute subgraph.
   *
   * @param graph Graph to iterate.
   * @param nodes_in_topological_order Subgraph nodes to recompute.
   * @param recompute_subgraph_output_node The final node of the subgraph.
   * @return Status
   */
  Status CreateRecomputeGraph(Graph& graph,
                              const InlinedVector<const Node*>& nodes_in_topological_order,
                              Node*& recompute_subgraph_output_node) const;

  /**************************************************
   ** Recompute related function definition ends   **
   *************************************************/

  // User enabled map of the subgraph string representation to the alleviation type.
  InlinedHashMap<std::string, optimizer::memory_optimizer::UserConfig> pattern_subgraph_to_user_optimizer_config_map_;
  std::string optimizer_config_;
  optimizer::memory_optimizer::ProbeLevel recompute_probe_level_;
  optimizer::memory_optimizer::ProbeConfig recompute_probe_config_;
};

}  // namespace onnxruntime
