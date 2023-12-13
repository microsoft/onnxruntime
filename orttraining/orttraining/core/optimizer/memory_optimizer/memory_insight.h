// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"

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
  class OutputStat {
   public:
    OutputStat(size_t output_index, std::string_view output_shape, size_t output_byte_count_per_element,
               float saving_ratio)
        : output_index(output_index),
          output_shape_str(output_shape),
          output_byte_count_per_element(output_byte_count_per_element),
          saving_ratio(saving_ratio) {}

    // output index, shape, byte count per element, saving ratio
    size_t output_index;
    std::string output_shape_str;
    size_t output_byte_count_per_element;
    float saving_ratio;
  };

  // Recompute Column
  std::string recompute_subgraph_str;
  InlinedVector<OutputStat> recomputed_outputs;
  int request_recompute_count = 0;
  int actual_recompute_count = 0;
  InlinedHashMap<size_t, int> output_port_reuse_recompute_count;

  // RecomputeWithCompromise Column
  std::string recompute_with_compromise_subgraph_str;
  InlinedVector<OutputStat> compromise_recomputed_outputs;
  int request_recompute_with_compromise_count = 0;
  int actual_recompute_with_compromise_count = 0;
  InlinedHashMap<size_t, int> output_port_reuse_recompute_with_compromise_count;

  // Frequency Column
  int freq = 0;
};

/**
 * @brief Reset `__backwardpass` attribute for all backward nodes in the graph.
 * `__backwardpass` is used by Priority-Based topology sorting.
 *
 * @param graph To be scanned and modified.
 * @param modified Whether the graph is modified.
 * @return Status
 */
Status ResetNodeBackwardPassAttribute(Graph& graph, bool& modified);

/**
 * @brief Iterate the graph and find all possible memory optimization opportunities for related nodes.
 *
 * @param graph_viewer  The graph to iterate.
 * @param probe_config The config for recomputable subgraph detecting.
 * @param logger Logger.
 * @param node_index_to_its_order_in_topological_sort_map  The mapping of node index to its order in topological sort.
 * @param yield_op_order_in_topological_sort The order of the boundary op in the topological sort.
 * @param candidate_output_args_map  A map from node to its candidate activations, which are consumed by both fw and
 * @param mem_opt_stats  A store to maintain all found optimization plans for related nodes.
 * @return Status
 */
Status FindORTModuleMemoryOpportunity(const GraphViewer& graph_viewer,
                                      const ProbeConfig& probe_config,
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
 * @param node_to_apply_contexts_map The optimization applying information.
 * @param generated_records Returns the generated memory optimization statistics table.
 * (for example, how many are actually applied) to each MemoryRecord.
 */
void GetMemoryRecordsGroupedByNodeClusterId(const MemoryOptimizationPlanner& memory_opt_planner,
                                            const NodeToClusterApplyContextMap&
                                                node_to_apply_contexts_map,
                                            std::vector<std::pair<std::string, MemoryRecord>>& generated_records);

/**
 * @brief Serialize the memory optimization statistics table to a string.
 *
 * @param records_grouped_by_node_cluster_id The memory optimization statistics table.
 * @param user_config The user configuration to the serialized string.
 * @return std::string
 */
std::string SerializeMemoryRecords(const std::vector<std::pair<std::string, MemoryRecord>>&
                                       records_grouped_by_node_cluster_id,
                                   std::string_view user_config);

/**
 * @brief A public API exposed to retrieve the memory optimization statistics table, given a graph.
 *
 * If possible, session's allocation plans and execution plan will also be available to help the analysis.
 *
 * @param graph_viewer The graph to analyze.
 * @param memory_optimization_config The user configuration to control the memory optimization.
 * @param recompute_probe_level The level to control allowed operations during recomputable subgraph detecting.
 * @param logger Logger.
 * @param ortvalue_name_to_idx_map Optional. If provided, we will use it to map ort value name to index.
 * @param p_seq_exec_plan Optional. If provided, we will use it to get allocation plans.
 * @return std::string
 */
std::string GetSerializedORTModuleMemoryStat(const GraphViewer& graph_viewer,
                                             std::string_view memory_optimization_config,
                                             std::string_view recompute_probe_level,
                                             const logging::Logger& logger,
                                             // used as Python binding, so used std::map instead of InlinedHashMap
                                             std::map<std::string, std::pair<std::string, int>>&
                                                 cluster_id_combinations_to_saved_symbolic_byte_map,
                                             const OrtValueNameIdxMap* ortvalue_name_to_idx_map = nullptr,
                                             const SequentialExecutionPlan* p_seq_exec_plan = nullptr);

}  // namespace onnxruntime::optimizer::memory_optimizer
