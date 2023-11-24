// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_insight.h"
#include "orttraining/core/optimizer/memory_optimizer/transformer_specific.h"

namespace onnxruntime::optimizer::memory_optimizer {

// Placeholder string for table row separator, which is used to be replaced by table row separator finally.
constexpr const char kTableRowSeparator[] = "TABLE_SEPARATOR_PLACEHOLDER";
// Placeholder string for table border, which is used to be replaced by table border finally.
constexpr const char kTableBorder[] = "TABLE_BORDER_PLACEHOLDER";

// The max length of the first column in the table.
constexpr const int kFirstColumnWidth = 7;
// The max length of left part (e.g. title) in the second column.
constexpr const int kTitleWidthInSecondColumn = 15;

/**
 * @brief Prepare info including activation usage, node usage in fw and bw.
 *
 * @param graph Graph to iterate.
 * @param boundary_op_order_in_topological_sort index of the boundary op between fw and bw.
 * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
 * @param fw_op_output_arg_used_map Collected activation usage mapping.
 *   - key: node arg name
 *   - value: a pair of bool, representing whether the activation is used by forward nodes or by backward nodes.
 * @param is_forward_nodes Collected node is forward pass op mapping.
 */
void GetForwardOutputUsageMap(const GraphViewer& graph_viewer,
                              const ptrdiff_t boundary_op_order_in_topological_sort,
                              const InlinedHashMap<NodeIndex, size_t>&
                                  node_index_to_its_order_in_topological_sort_map,
                              ActivationUsedMap& fw_op_output_arg_used_map,
                              InlinedHashMap<const Node*, bool>& is_forward_nodes) {
  ORT_ENFORCE(boundary_op_order_in_topological_sort >= 0);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();
  is_forward_nodes.clear();
  is_forward_nodes.reserve(node_ids.size());

  auto is_forward_pass_operator = [](ptrdiff_t op_order_in_topological_sort,
                                     ptrdiff_t boundary_op_order_in_topological_sort) -> bool {
    return op_order_in_topological_sort <= boundary_op_order_in_topological_sort;
  };

  fw_op_output_arg_used_map.clear();
  fw_op_output_arg_used_map.reserve(node_ids.size());
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr /* skip removed nodes*/) {
      continue;
    }

    const Node& node = *p_node;

    bool is_forward_op = is_forward_pass_operator(static_cast<ptrdiff_t>(i), boundary_op_order_in_topological_sort);
    if (!is_forward_op) {
      is_forward_nodes[p_node] = false;
      continue;
    }

    is_forward_nodes[p_node] = true;

    for (auto& output_arg : node.OutputDefs()) {
      if (!output_arg->Exists() || output_arg->Name().empty()) {
        continue;
      }

      bool used_in_fw = false;
      bool used_in_bw = false;
      for (auto& consumer_node : graph_viewer.GetConsumerNodes(output_arg->Name())) {
        ORT_ENFORCE(consumer_node != nullptr, "Consumer node should not be null.");
        auto it = node_index_to_its_order_in_topological_sort_map.find(consumer_node->Index());
        ORT_ENFORCE(it !=
                        node_index_to_its_order_in_topological_sort_map.end(),
                    "Consumer node should be in topological order map.");
        size_t consumer_node_index_in_topological_order = it->second;
        if (is_forward_pass_operator(static_cast<ptrdiff_t>(consumer_node_index_in_topological_order),
                                     boundary_op_order_in_topological_sort)) {
          used_in_fw = true;
        } else {
          used_in_bw = true;
        }
      }

      ORT_ENFORCE(fw_op_output_arg_used_map.find(output_arg->Name()) == fw_op_output_arg_used_map.end(),
                  "Duplicated output arg found named: ", output_arg->Name());
      fw_op_output_arg_used_map.insert({{output_arg->Name(), std::make_pair(used_in_fw, used_in_bw)}});
    }
  }
}

/**
 * @brief Find all stashed activations, e.g. activations used by forward operators and backward operators.
 *
 * @param graph_viewer Graph to iterate.
 * @param boundary_op_order_in_topological_sort The order of the boundary op in the topological sort.
 * @param fw_op_output_arg_used_map Activation usage mapping.
 * @param candidate_output_args_map Candidate activations, which are consumed by both fw and bw ops.
 * @param is_forward_nodes Whether a node is a forward node.
 * @param logger Logger.
 * @return Status
 */

Status GetStashedActivationCandidates(const GraphViewer& graph_viewer,
                                      const ptrdiff_t boundary_op_order_in_topological_sort,
                                      ActivationUsedMap& fw_op_output_arg_used_map,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                          candidate_output_args_map,
                                      InlinedHashMap<const Node*, bool>& is_forward_nodes,
                                      const logging::Logger& logger) {
  if (boundary_op_order_in_topological_sort < 0) {
    LOGS(logger, VERBOSE) << "No boundary op found. Skip memory optimization.";
    return Status::OK();
  }

  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  InlinedHashMap<NodeIndex, size_t> node_index_to_its_order_in_topological_sort_map;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = i;
  }

  GetForwardOutputUsageMap(graph_viewer, boundary_op_order_in_topological_sort,
                           node_index_to_its_order_in_topological_sort_map,
                           fw_op_output_arg_used_map,
                           is_forward_nodes);

  for (auto& kv : fw_op_output_arg_used_map) {
    // used by fw and bw, then it is a candidate.
    if (kv.second.first && kv.second.second) {
      const Node* n = graph_viewer.GetProducerNode(kv.first);
      ORT_ENFORCE(n, "Activation should have a producer node");
      size_t k = 0;
      for (k = 0; k < n->OutputDefs().size(); ++k) {
        if (n->OutputDefs()[k]->Name().compare(kv.first) == 0) {
          break;
        }
      }

      if (std::find(candidate_output_args_map[n].begin(), candidate_output_args_map[n].end(), k) !=
          candidate_output_args_map[n].end()) {
        ORT_ENFORCE(false, "Duplicated candidate output found.");
      }

      candidate_output_args_map[n].push_back(k);
      LOGS(logger, VERBOSE) << "Find candidate output named [" << kv.first << "] of Node " << n->Name() << "("
                            << n->OpType() << ")";
    }
  }

  return Status::OK();
}

Status FindORTModuleMemoryOpportunity(const GraphViewer& graph_viewer,
                                      const ProbeLevel probe_level,
                                      const logging::Logger& logger,
                                      InlinedHashMap<NodeIndex, ptrdiff_t>&
                                          node_index_to_its_order_in_topological_sort_map,
                                      ptrdiff_t& yield_op_order_in_topological_sort,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                          candidate_output_args_map,
                                      MemoryOptimizationPlanner& memory_opt_planner) {
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Find boundary ops between forward and backward pass, currently, it's limited to YieldOp.
  yield_op_order_in_topological_sort = -1;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    if (p_node->OpType() == "YieldOp") {
      if (yield_op_order_in_topological_sort != -1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "There are multiple YieldOps in the graph, node: ",
                               p_node->Name(), " is the second one.");
      }
      yield_op_order_in_topological_sort = static_cast<ptrdiff_t>(i);
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = static_cast<ptrdiff_t>(i);
  }

  ActivationUsedMap fw_op_output_arg_used_map;

  InlinedHashMap<const Node*, bool> is_forward_nodes;
  ORT_RETURN_IF_ERROR(GetStashedActivationCandidates(graph_viewer,
                                                     yield_op_order_in_topological_sort,
                                                     fw_op_output_arg_used_map,
                                                     candidate_output_args_map,
                                                     is_forward_nodes,
                                                     logger));

  InlinedHashSet<const Node*> layer_boundary_ln_nodes;
  FindLayerBoundaryLayerNodeNodes(graph_viewer, logger, layer_boundary_ln_nodes);

  // The first pass - find the candidate subgraphs.
  for (int i = static_cast<int>(node_ids.size()) - 1; i >= 0; --i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    if (candidate_output_args_map.find(p_node) == candidate_output_args_map.end()) {
      continue;
    }

    bool can_compromise_stashed_activation = false;
    std::unique_ptr<NodeRecomputePlan> recompute_plan =
        CheckNodeForRecompute(graph_viewer,
                              *p_node,
                              probe_level,
                              fw_op_output_arg_used_map,
                              node_index_to_its_order_in_topological_sort_map,
                              candidate_output_args_map,
                              layer_boundary_ln_nodes,
                              logger, false,
                              can_compromise_stashed_activation);
    if (recompute_plan != nullptr) {
      memory_opt_planner.AddNodeOptimizationPlan(p_node, std::move(recompute_plan));
    }

    if (can_compromise_stashed_activation) {
      LOGS(logger, VERBOSE) << "Searching Node " << p_node->Name() << "(" << p_node->OpType()
                            << ") for compromised recompute";
      // If the subgraph recompute can save memory by comprising the assumption - recompute graphs' input must exist
      // during backward pass, then we can consider to recompute them.
      std::unique_ptr<NodeRecomputePlan> recompute_with_compromise_plan =
          CheckNodeForRecompute(graph_viewer, *p_node, probe_level, fw_op_output_arg_used_map,
                                node_index_to_its_order_in_topological_sort_map,
                                candidate_output_args_map,
                                layer_boundary_ln_nodes,
                                logger, true,
                                can_compromise_stashed_activation);
      if (recompute_with_compromise_plan != nullptr) {
        memory_opt_planner.AddNodeOptimizationPlan(p_node, std::move(recompute_with_compromise_plan));
      }
    }
  }

  return Status::OK();
}

void GetMemoryRecordsGroupedByNodeClusterId(const MemoryOptimizationPlanner& memory_opt_planner,
                                            const NodeToClusterApplyContextMap& node_to_apply_contexts_map,
                                            std::vector<std::pair<std::string, MemoryRecord>>& generated_records) {
  // Group by node cluster id, generate memory record.
  InlinedHashMap<std::string, MemoryRecord> records;
  const auto& node_to_optimization_plan_map = memory_opt_planner.GetNodeToOptimizationPlanMap();
  for (const auto& node_to_optimization_plan : node_to_optimization_plan_map) {
    const auto& node = node_to_optimization_plan.first;
    const auto& node_plans = node_to_optimization_plan.second;
    const std::string node_cluster_id = memory_opt_planner.GenerateNodeClusterId(node);

    std::pair<InlinedHashMap<std::string, MemoryRecord>::iterator, bool> insert_result =
        records.insert({node_cluster_id, MemoryRecord()});
    bool already_exist = !insert_result.second;
    auto& record = insert_result.first->second;
    record.freq++;

    // Collect more information for display.
    for (auto& plan : node_plans) {
      // Same node cluster id, plans might still have different reuse_buffer pattern, so we need to collect all of them.
      if (plan->reuse_buffers.size() > 0) {
        gsl::span<const size_t> output_indices = plan->GetActivationOutputIndices();
        for (auto output_index : output_indices) {
          bool is_output_reusing_buffers = plan->reuse_buffers.find(output_index) != plan->reuse_buffers.end();
          if (plan->GetOptimizationType() == OptimizationType::RecomputeWithCompromise) {
            if (is_output_reusing_buffers) {
              record.output_port_reuse_recompute_with_compromise_count[output_index] += 1;
            }
          } else if (plan->GetOptimizationType() == OptimizationType::Recompute) {
            if (is_output_reusing_buffers) {
              record.output_port_reuse_recompute_count[output_index] += 1;
            }
          }
        }
      }

      // For other infos that are guaranteed identity by cluster id, just skip collecting.
      if (already_exist) {
        continue;
      }

      if (plan->GetOptimizationType() == OptimizationType::RecomputeWithCompromise) {
        record.recompute_with_compromise_subgraph_str =
            dynamic_cast<NodeRecomputePlan*>(plan.get())->GetNodesInTopoOrderStr();
      } else if (plan->GetOptimizationType() == OptimizationType::Recompute) {
        record.recompute_subgraph_str = dynamic_cast<NodeRecomputePlan*>(plan.get())->GetNodesInTopoOrderStr();
      }

      gsl::span<const size_t> output_indices = plan->GetActivationOutputIndices();
      for (auto output_index : output_indices) {
        const auto& output_def = node->OutputDefs()[output_index];
        MLDataType ml_data_type = DataTypeImpl::TypeFromProto(*output_def->TypeAsProto());
        ORT_ENFORCE(ml_data_type->IsTensorType(), "ml_type must be a tensor type, but it is ",
                    DataTypeImpl::ToString(ml_data_type));
        const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
        ORT_ENFORCE(nullptr != tensor_type_base);
        MLDataType elt_type = tensor_type_base->GetElementType();

        const auto byte_count_per_element = elt_type->Size();
        if (plan->GetOptimizationType() == OptimizationType::RecomputeWithCompromise) {
          record.compromise_recomputed_outputs.emplace_back(
              output_index,
              GetTensorElemCountInSymbolicString(node, output_index),
              byte_count_per_element,
              plan->GetSaveRatio());

        } else if (plan->GetOptimizationType() == OptimizationType::Recompute) {
          record.recomputed_outputs.emplace_back(output_index,
                                                 GetTensorElemCountInSymbolicString(node, output_index),
                                                 byte_count_per_element,
                                                 plan->GetSaveRatio());
        }
      }
    }
  }

  // Sort by feq and then by record key, to make sure the output is deterministic.
  InlinedVector<std::pair<int, std::string>> freq_to_record_key;
  for (const auto& p : records) {
    freq_to_record_key.push_back({p.second.freq, p.first});
  }

  std::sort(freq_to_record_key.begin(), freq_to_record_key.end(), [](auto& left, auto& right) {
    if (left.first == right.first) {
      return left.second.compare(right.second) > 0;
    }
    return left.first > right.first;
  });

  for (const auto& p : freq_to_record_key) {
    const std::string record_key = p.second;
    generated_records.push_back({record_key, records[record_key]});
  }

  // If apply context is provided, also update the actual applied count.
  if (node_to_apply_contexts_map.size() > 0) {
    InlinedHashMap<std::string, MemoryRecord*> node_cluster_id_to_record_map;
    for (auto& p : generated_records) {
      node_cluster_id_to_record_map[p.first] = &p.second;
    }

    for (const auto& p : node_to_apply_contexts_map) {
      const auto& node = p.first;
      const auto& apply_context = p.second;
      std::string node_cluster_id = memory_opt_planner.GenerateNodeClusterId(node);
      if (apply_context->type == OptimizationType::Recompute) {
        node_cluster_id_to_record_map[node_cluster_id]->actual_recompute_count += 1;
        node_cluster_id_to_record_map[node_cluster_id]->request_recompute_count = apply_context->requested_count;
      } else if (apply_context->type == OptimizationType::RecomputeWithCompromise) {
        node_cluster_id_to_record_map[node_cluster_id]->actual_recompute_with_compromise_count += 1;
        node_cluster_id_to_record_map[node_cluster_id]->request_recompute_with_compromise_count =
            apply_context->requested_count;
      } else {
        ORT_THROW("Unsupported optimization type found.");
      }
    }
  }
}

// Function declare to make it compile.
void IterateNodeOptimizationPlan(const std::shared_ptr<NodeOptimizationPlanBase>& plan,
                                 const InlinedHashMap<const Node*, InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                                     node_to_optimization_plans_map,
                                 const InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>&
                                     current_combination,
                                 const logging::Logger& logger,
                                 InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                                     all_combinations);

/*
 * Iterate from a node, generate combinations for each optimization plan for it.
 */
void IterateNode(const Node* node,
                 const InlinedHashMap<const Node*, InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                     node_to_optimization_plans_map,
                 const InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>&
                     current_combination,
                 const logging::Logger& logger,
                 InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                     all_combinations) {
  MO_LOG_DEBUG_INFO(logger, "Enter IterateNode: " + node->Name());
  if (node_to_optimization_plans_map.find(node) == node_to_optimization_plans_map.end()) {
    MO_LOG_DEBUG_INFO(logger, "Exit IterateNode since reused node don't have optimization plans: " + node->Name());
    return;
  }

  for (const std::shared_ptr<NodeOptimizationPlanBase>& plan : node_to_optimization_plans_map.at(node)) {
    if (std::find(current_combination.begin(), current_combination.end(), plan) !=
        current_combination.end()) {
      continue;
    }
    InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>> new_combination = current_combination;
    new_combination.push_back(plan);
    IterateNodeOptimizationPlan(plan, node_to_optimization_plans_map, new_combination, logger, all_combinations);
  }
  MO_LOG_DEBUG_INFO(logger, "Exit IterateNode: " + node->Name());
}

void ListAllCombinations(const InlinedVector<InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>>&
                             all_possible_node_optimization_plans,
                         int index,
                         const InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>& current_combination,
                         const logging::Logger& logger,
                         InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                             all_combinations) {
  MO_LOG_DEBUG_INFO(logger, "Enter ListAllCombinations");
  if (index == static_cast<int>(all_possible_node_optimization_plans.size())) {
    if (std::find(all_combinations.begin(), all_combinations.end(), current_combination) ==
        all_combinations.end()) {
      all_combinations.push_back(current_combination);
    }
    MO_LOG_DEBUG_INFO(logger, "Exit ListAllCombinations after finding a new combination");
    return;
  }

  for (const auto& plans : all_possible_node_optimization_plans[index]) {
    for (const auto& plan : plans) {
      InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>> new_combination = current_combination;
      new_combination.push_back(plan);
      ListAllCombinations(all_possible_node_optimization_plans, index + 1, new_combination, logger, all_combinations);
    }
  }

  MO_LOG_DEBUG_INFO(logger, "Exit ListAllCombinations");
}

/**
 * Iterate from a node optimization plan, if there is any buffer reuse in its node outputs,
 * iterate all possible reuse buffer plan combinations.
 */
void IterateNodeOptimizationPlan(const std::shared_ptr<NodeOptimizationPlanBase>& plan,
                                 const InlinedHashMap<const Node*, InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                                     node_to_optimization_plans_map,
                                 const InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>&
                                     current_combination,
                                 const logging::Logger& logger,
                                 InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
                                     all_combinations) {
  MO_LOG_DEBUG_INFO(logger, "Enter IterateNodeOptimizationPlan: " + plan->GetClusterId());

  // No reuse buffer, don't need to iterate further, we found a plan combination already.
  if (plan->reuse_buffers.size() == 0) {
    MO_LOG_DEBUG_INFO(logger, "length of current_combination: " +
                                  std::to_string(current_combination.size()) + ", " + plan->GetClusterId());
    all_combinations.push_back(current_combination);
    MO_LOG_DEBUG_INFO(logger, "Exit IterateNodeOptimizationPlan");
    return;
  }

  InlinedVector<InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>>
      all_possible_node_optimization_plans;
  all_possible_node_optimization_plans.resize(plan->reuse_buffers.size());

  size_t i = 0;
  for (const auto& p : plan->reuse_buffers) {
    MO_LOG_DEBUG_INFO(logger, ">>>reuse buffer: " + std::to_string(p.first));
    IterateNode(p.second.first, node_to_optimization_plans_map, {}, logger, all_possible_node_optimization_plans[i]);
    ++i;
  }

  ListAllCombinations(all_possible_node_optimization_plans, 0, current_combination, logger, all_combinations);

  MO_LOG_DEBUG_INFO(logger, "Exit IterateNodeOptimizationPlan: " + plan->GetClusterId());
}

// Return a deterministic string for multiple plans combinations.
std::string GetMultiplePlanClusterId(const InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>& plans) {
  constexpr const int request_count = -1;  // -1 means apply optimization to all appearances.

  std::ostringstream oss;
  InlinedVector<std::string> sorted_plans;
  for (const auto& plan : plans) {
    sorted_plans.push_back(plan->GetClusterId() + ":" + std::to_string(static_cast<int>(plan->GetOptimizationType())) +
                           ":" + std::to_string(request_count));
  }

  std::sort(sorted_plans.begin(), sorted_plans.end());

  for (const auto& plan : sorted_plans) {
    if (oss.str().size() > 0) {
      oss << ",";
    }
    oss << plan;
  }
  return oss.str();
}

void GetMemorySavingSymbolicString(const MemoryOptimizationPlanner& memory_opt_planner,
                                   const logging::Logger& logger,
                                   std::map<std::string, std::pair<std::string, int>>&
                                       combination_cluster_ids_to_saved_symbolic_byte_map) {
  // Group by "ClusterId:OptimizationType:RequestCount".
  InlinedVector<InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>> all_combinations;

  combination_cluster_ids_to_saved_symbolic_byte_map.clear();
  const auto& node_to_optimization_plan_map = memory_opt_planner.GetNodeToOptimizationPlanMap();
  for (const auto& node_to_optimization_plan : node_to_optimization_plan_map) {
    const auto& node = node_to_optimization_plan.first;
    InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>> current_combination;
    MO_LOG_DEBUG_INFO(logger, ">>>Start looping node: " + node->Name());
    IterateNode(node, node_to_optimization_plan_map, current_combination, logger, all_combinations);
    MO_LOG_DEBUG_INFO(logger, "<<<End looping node: " + node->Name());
  }

  for (const auto& combination : all_combinations) {
    std::string combination_cluster_id = GetMultiplePlanClusterId(combination);
    std::string symbolic_byte_count = "";
    for (const auto& plan : combination) {
      if (symbolic_byte_count.size() > 0) {
        symbolic_byte_count += " + ";
      }
      symbolic_byte_count += plan->GetMemorySavingSymbolicString();
    }

    if (symbolic_byte_count.size() > 0) {
      symbolic_byte_count = "(" + symbolic_byte_count + ")";
    }
    auto& p = combination_cluster_ids_to_saved_symbolic_byte_map[combination_cluster_id];
    const auto& original = p.first;
    if (original.size() > 0) {
      symbolic_byte_count = original + " + " + symbolic_byte_count;
    }

    MO_LOG_DEBUG_INFO(logger, "combination_cluster_id: " + combination_cluster_id +
                                  ", symbolic_byte_count: " + symbolic_byte_count);

    p.first = symbolic_byte_count;
    p.second += 1;
  }
}

namespace {

template <typename T>
std::string ToFixedLengthString(T value, int length) {
  std::ostringstream oss;
  oss << std::setw(length) << std::left;
  oss << value;
  return oss.str();
}

void FormatRecomputeMemoryRecords(int option_index,
                                  const MemoryRecord& record,
                                  bool compromise_recompute,
                                  InlinedVector<std::string>& rows) {
  const auto subgraph_str = compromise_recompute ? record.recompute_with_compromise_subgraph_str
                                                 : record.recompute_subgraph_str;
  const auto opt_type = compromise_recompute ? OptimizationType::RecomputeWithCompromise
                                             : OptimizationType::Recompute;
  const auto request_count = compromise_recompute ? record.request_recompute_with_compromise_count
                                                  : record.request_recompute_count;
  const auto actual_count = compromise_recompute ? record.actual_recompute_with_compromise_count
                                                 : record.actual_recompute_count;

  const std::string empty_first_col = "|" + ToFixedLengthString(std::string(), kFirstColumnWidth) + "|";

  rows.push_back(empty_first_col);
  rows.push_back(empty_first_col +
                 ToFixedLengthString(">>Option " + std::to_string(option_index), kTitleWidthInSecondColumn) + ": " +
                 OptimizationTypeToString(opt_type) + " subgraph " + subgraph_str);

  if (request_count) {
    // Only show this if user requested it.
    rows.push_back(
        empty_first_col +
        ToFixedLengthString("  Status", kTitleWidthInSecondColumn) + ": " + "Enabled, requested count=" +
        std::to_string(request_count) +
        ", actual applied count=" + std::to_string(actual_count));
  } else {
    rows.push_back(empty_first_col + ToFixedLengthString("  Status", kTitleWidthInSecondColumn) +
                   ": Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=" +
                   subgraph_str + ":" + std::to_string(static_cast<int>(opt_type)) + ":-1");
  }

  std::string activation_str = empty_first_col + "  Stashed Activations: ";
  rows.push_back(activation_str);

  const auto& reused_buffers = compromise_recompute ? record.output_port_reuse_recompute_with_compromise_count
                                                    : record.output_port_reuse_recompute_count;
  if (reused_buffers.size() > 0) {
    std::string reused_buffers_summary = empty_first_col + ToFixedLengthString("   - ReuseFreq", kTitleWidthInSecondColumn) + ": ";
    for (const auto& p : reused_buffers) {
      reused_buffers_summary += " Output " + std::to_string(p.first) + "(" + std::to_string(p.second) + "),";
    }

    rows.push_back(reused_buffers_summary);
  }

  const auto activation_count = compromise_recompute ? record.compromise_recomputed_outputs.size()
                                                     : record.recomputed_outputs.size();
  for (size_t i = 0; i < activation_count; ++i) {
    const MemoryRecord::OutputStat* stat;
    if (compromise_recompute) {
      stat = &record.compromise_recomputed_outputs[i];
    } else {
      stat = &record.recomputed_outputs[i];
    }

    rows.push_back(empty_first_col +
                   ToFixedLengthString("   - Output " + std::to_string(stat->output_index), kTitleWidthInSecondColumn) +
                   ": [" + stat->output_shape_str + "], byte/elem: " +
                   std::to_string(stat->output_byte_count_per_element) +
                   ", " + std::to_string(static_cast<int>(stat->saving_ratio * 100)) +
                   "% saved");
  }
}
}  // namespace

std::string SerializeMemoryRecords(
    const std::vector<std::pair<std::string, MemoryRecord>>& records_grouped_by_node_cluster_id,
    std::string_view user_config) {
  InlinedVector<std::string> rows;
  rows.push_back(kTableBorder);
  rows.push_back("|" + ToFixedLengthString("Freq", kFirstColumnWidth) +
                 "| Memory Optimization Opportunities (Clustered by node-level activation patterns)");
  rows.push_back(kTableRowSeparator);

  for (const auto& p : records_grouped_by_node_cluster_id) {
    const auto& record = p.second;
    rows.push_back("|" + ToFixedLengthString(record.freq, kFirstColumnWidth) +
                   "|For each row options are mutually exclusive, only one of them can be enabled.");

    int option_index = 1;
    if (record.recomputed_outputs.size() > 0) {
      FormatRecomputeMemoryRecords(option_index, record, false, rows);
      option_index++;
    }

    if (record.compromise_recomputed_outputs.size() > 0) {
      FormatRecomputeMemoryRecords(option_index, record, true, rows);
      option_index++;
    }
    rows.push_back(kTableRowSeparator);
  }

  rows.push_back(kTableBorder);

  size_t max_length = 0;
  for (auto& row : rows) {
    max_length = std::max(max_length, row.length());
  }

  // Example is:
  // static const std::string row_separator =
  //     "|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|\n";
  static const std::string kTableRowSeparatorStart = "|_ _ _ _|";
  size_t second_row_length = max_length - kTableRowSeparatorStart.length();
  if (second_row_length % 2 == 0) {
    second_row_length += 2;
    max_length += 2;
  } else {
    second_row_length += 3;  // add 3 to make it even
    max_length += 3;
  }
  std::string row_separator_full(second_row_length, ' ');
  for (size_t i = 0; i < row_separator_full.size() - 1; ++i) {
    if (i % 2 == 0) {
      row_separator_full[i] = '_';
    }
  }
  row_separator_full[row_separator_full.size() - 1] = '|';
  row_separator_full = kTableRowSeparatorStart + row_separator_full;

  std::string table_border_full(max_length, '=');
  std::ostringstream summary;
  summary << std::endl;
  summary << MakeString("MemoryInsight Summary - User config: ", (user_config.empty() ? "not provided" : user_config))
          << std::endl;
  for (auto& row : rows) {
    if (row == kTableRowSeparator) {
      summary << row_separator_full << std::endl;
    } else if (row == kTableBorder) {
      summary << table_border_full << std::endl;
    } else {
      std::string filled_up = std::string(max_length - row.length(), ' ');
      filled_up[filled_up.length() - 1] = '|';
      summary << row << filled_up << std::endl;
    }
  }
  summary << "Note: use comma as a separator for enabling more than one subgraphs." << std::endl;
  return summary.str();
}

std::string GetSerializedORTModuleMemoryStat(const GraphViewer& graph_viewer,
                                             std::string_view memory_optimization_config,
                                             std::string_view recompute_probe_level,
                                             const logging::Logger& logger,
                                             std::map<std::string, std::pair<std::string, int>>&
                                                 cluster_id_combinations_to_saved_symbolic_byte_map,
                                             const OrtValueNameIdxMap* ortvalue_name_to_idx_map,
                                             const SequentialExecutionPlan* p_seq_exec_plan) {
  ProbeLevel probe_level = ProbeLevel::Advanced;
  if (!recompute_probe_level.empty()) {
    int probe_level_int = ParseIntValueFromString(recompute_probe_level);
    ORT_ENFORCE(probe_level_int < static_cast<int>(ProbeLevel::LevelMax) &&
                    probe_level_int >= 0,
                "Invalid probe level specified: ", recompute_probe_level);
    probe_level = static_cast<ProbeLevel>(probe_level_int);
  }

  ptrdiff_t yield_op_order_in_topological_sort;
  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  InlinedHashMap<NodeIndex, ptrdiff_t> node_index_to_its_order_in_topological_sort_map;

  // The first pass - find the candidate subgraphs.
  MemoryOptimizationPlanner memory_opt_planner;
  ORT_ENFORCE(FindORTModuleMemoryOpportunity(
                  graph_viewer,
                  probe_level,
                  logger,
                  node_index_to_its_order_in_topological_sort_map,
                  yield_op_order_in_topological_sort,
                  candidate_output_args_map,
                  memory_opt_planner)
                  .IsOK());

  InlinedHashMap<std::string, UserConfig> cluster_id_to_config_map;
  // Finalize the plan according to user config,
  // then create a ClusterApplyContext for each unique cluster (having the same node pattern)

  NodeToClusterApplyContextMap node_to_apply_context_map;

  if (!memory_optimization_config.empty()) {
    ORT_ENFORCE(ParseConfigFromString(memory_optimization_config, cluster_id_to_config_map)
                    .IsOK());
    InlinedHashMap<const Node*, std::shared_ptr<NodeOptimizationPlanBase>> node_to_opt_plan_map;
    ORT_ENFORCE(memory_opt_planner.FinalizeNodePlansFromUserConfig(cluster_id_to_config_map,
                                                                   node_to_opt_plan_map,
                                                                   node_to_apply_context_map)
                    .IsOK());
  }

  if (ortvalue_name_to_idx_map != nullptr && p_seq_exec_plan != nullptr) {
    ORT_ENFORCE(memory_opt_planner.UpdateNodePlansFromExecutionPlan(graph_viewer,
                                                                    *ortvalue_name_to_idx_map,
                                                                    *p_seq_exec_plan)
                    .IsOK());
  }

  std::vector<std::pair<std::string, MemoryRecord>> records;
  GetMemoryRecordsGroupedByNodeClusterId(memory_opt_planner, node_to_apply_context_map, records);

  GetMemorySavingSymbolicString(memory_opt_planner, logger, cluster_id_combinations_to_saved_symbolic_byte_map);

  return SerializeMemoryRecords(records, memory_optimization_config);
}

}  // namespace onnxruntime::optimizer::memory_optimizer
