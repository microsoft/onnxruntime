// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"

// #include "core/optimizer/utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

/**
 * @brief A data structure to store memory optimization statistics for a specific node cluster id.
 *
 * We will collect statistics for each node cluster id.
 * The node cluster id is generated from all possible optimization plans for a specific node, plus shape, data type,
 * outputs, etc. For the nodes have the same node cluster id, they will have one single MemoryRecord, displayed
 * as a row in the final memory optimization statistics table.
 */
class MemoryRecord {
 public:
  // Recompute Column
  std::string recompute_subgraph_str;
  /// output index, shape, byte count per element
  std::vector<std::tuple<size_t, std::string, int>> recomputed_outputs;
  int request_recompute_count = 0;
  int actual_recompute_count = 0;
  std::unordered_map<size_t, int> output_port_reuse_recompute_count;

  // RecomputeWithCompromise Column
  std::string recompute_with_compromise_subgraph_str;
  /// output index, shape, byte count per element, saving ratio
  std::vector<std::tuple<size_t, std::string, int, float>> compromise_recomputed_outputs;
  int request_recompute_with_compromise_count = 0;
  int actual_recompute_with_compromise_count = 0;
  std::unordered_map<size_t, int> output_port_reuse_recompute_with_compromise_count;

  // Frequency Column
  int freq = 0;
};

/**
 * @brief Iterate the graph and find all possible memory optimization opportunities for related nodes.
 *
 * @param graph  The graph to iterate.
 * @param probe_level The level to control allowed operations during recomputable subgraph detecting.
 * @param logger Logger.
 * @param node_index_to_its_order_in_topological_sort_map  The mapping of node index to its order in topological sort.
 * @param yield_op_order_in_topological_sort The order of the boundary op in the topological sort.
 * @param candidate_output_args_map  A map from node to its candidate activations, which are consumed by both fw and
 * @param mem_opt_stats  A store to maintain all found optimization plans for related nodes.
 * @return Status
 */
Status FindORTModuleMemoryOpportunity(const Graph& graph,
                                      const ProbeLevel probe_level,
                                      const logging::Logger& logger,
                                      InlinedHashMap<NodeIndex, ptrdiff_t>&
                                          node_index_to_its_order_in_topological_sort_map,
                                      ptrdiff_t& yield_op_order_in_topological_sort,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map,
                                      MemoryOptimizationPlanner& mem_opt_stats);

/**
 * @brief From the optimization plans, generate the memory optimization statistics table containing many MemoryRecords,
 * each represents one node cluster id.
 *
 * @param memory_opt_planner The optimization planner to get optimization plans.
 * @param generated_records Returns the generated memory optimization statistics table.
 * @param node_to_apply_contexts_map Optional. If provided, we will append the optimization applying information
 * (for example, how many are actually applied) to each MemoryRecord.
 */
void GetMemoryRecordsGroupedByNodeClusterId(const MemoryOptimizationPlanner& memory_opt_planner,
                                            std::vector<std::pair<std::string, MemoryRecord>>& generated_records,
                                            const InlinedHashMap<const Node*, std::shared_ptr<ClusterApplyContext>>&
                                                node_to_apply_contexts_map);

/**
 * @brief Serialize the memory optimization statistics table to a string.
 *
 * @param records_grouped_by_node_cluster_id The memory optimization statistics table.
 * @param user_config Optional. If provided, we will append the user configuration to the serialized string.
 * @return std::string
 */
std::string SerializeMemoryRecords(std::vector<std::pair<std::string, optimizer::memory_optimizer::MemoryRecord>>
                                       records_grouped_by_node_cluster_id,
                                   const std::string user_config = "");

/**
 * @brief A public API exposed to retrieve the memory optimization statistics table, given a graph.
 *
 * If possible, session's allocation plans and execution plan will also be available to help the analysis.
 *
 * @param graph The graph to analyze.
 * @param memory_optimization_config The user configuration to control the memory optimization.
 * @param recompute_probe_level The level to control allowed operations during recomputable subgraph detecting.
 * @param logger Logger.
 * @param ortvalue_name_to_idx_map Optional. If provided, we will use it to map ort value name to index.
 * @param p_seq_exec_plan Optional. If provided, we will use it to get allocation plans.
 * @return std::string
 */
std::string GetSerializedORTModuleMemoryStat(const Graph& graph,
                                             const std::string& memory_optimization_config,
                                             const std::string recompute_probe_level,
                                             const logging::Logger& logger,
                                             const OrtValueNameIdxMap& ortvalue_name_to_idx_map = {},
                                             const SequentialExecutionPlan& p_seq_exec_plan = {});
}  // namespace onnxruntime::optimizer::memory_optimizer
