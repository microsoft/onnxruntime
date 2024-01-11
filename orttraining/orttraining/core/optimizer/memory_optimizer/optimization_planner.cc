// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>

#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"

namespace onnxruntime::optimizer::memory_optimizer {

Status MemoryOptimizationPlanner::UpdateNodePlansFromExecutionPlan(const GraphViewer& graph_viewer,
                                                                   const OrtValueNameIdxMap& ortvalue_name_to_idx_map,
                                                                   const SequentialExecutionPlan& p_seq_exec_plan) {
  InlinedHashMap<int, std::string> idx_to_ortvalue_name_map;
  for (const auto& entry : ortvalue_name_to_idx_map) {
    idx_to_ortvalue_name_map[entry.second] = entry.first;
  }

  for (const auto& node_to_optimization_plan : node_to_optimization_plans_map) {
    const auto& node_plans = node_to_optimization_plan.second;

    for (auto& node_plan : node_plans) {
      const std::string cluster_id = node_plan->GetClusterId();
      const Node* node = node_plan->node;
      for (auto& output_index : node_plan->GetActivationOutputIndices()) {
        const NodeArg* node_arg = node->OutputDefs()[output_index];
        const auto& ort_value_name = node_arg->Name();
        int ort_value_idx;
        ORT_ENFORCE(ortvalue_name_to_idx_map.GetIdx(ort_value_name, ort_value_idx).IsOK());
        const auto& alloc_plan = p_seq_exec_plan.allocation_plan;
        ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
        const auto& per_alloc_plan = alloc_plan[ort_value_idx];
        if (per_alloc_plan.alloc_kind != AllocKind::kReuse) {
          continue;
        }
        int reused_ort_value_idx = per_alloc_plan.reused_buffer;
        const auto& reused_ort_value_name = idx_to_ortvalue_name_map.at(reused_ort_value_idx);

        const Node* p_node = graph_viewer.GetProducerNode(reused_ort_value_name);
        if (p_node == nullptr) {
          // This is a graph input.
          continue;
        }

        int src_op_output_index = optimizer_utils::IndexOfNodeOutput(*p_node, *node_arg);
        node_plan->reuse_buffers[output_index] = std::make_pair(p_node, src_op_output_index);
      }
    }
  }

  return Status::OK();
}

Status MemoryOptimizationPlanner::FinalizeNodePlansFromUserConfig(
    const InlinedHashMap<std::string, UserConfig>& cluster_id_to_user_configs,
    InlinedHashMap<const Node*, std::shared_ptr<NodeOptimizationPlanBase>>& node_to_opt_plan_map,
    NodeToClusterApplyContextMap& node_to_apply_context_map) const {
  if (cluster_id_to_user_configs.size() == 0) {
    return Status::OK();
  }

  // Create a temporary map to store the apply context for each cluster pattern.
  InlinedHashMap<std::string, std::shared_ptr<ClusterApplyContext>> cluster_id_to_apply_contexts_map;

  // We loop all nodes' optimization plans and find the match in user configs.
  // If found in user configs, we finalize the plan and create/update the apply context for this node.
  // If not found in user configs, we will not include the node in the returned result.
  for (const auto& node_to_optimization_plan : node_to_optimization_plans_map) {
    const auto& node = node_to_optimization_plan.first;
    const auto& node_plans = node_to_optimization_plan.second;

    for (auto& node_plan : node_plans) {
      const std::string cluster_id = node_plan->GetClusterId();
      if (cluster_id_to_user_configs.find(cluster_id) == cluster_id_to_user_configs.end()) {
        continue;
      }

      const auto& user_config = cluster_id_to_user_configs.at(cluster_id);
      if (node_plan->GetOptimizationType() == user_config.type) {
        // First finalize the plan for this node.
        node_to_opt_plan_map[node] = node_plan;

        // Create/Update the apply context for this node.
        if (cluster_id_to_apply_contexts_map.find(cluster_id) == cluster_id_to_apply_contexts_map.end()) {
          std::shared_ptr<ClusterApplyContext> apply_context = std::make_shared<ClusterApplyContext>();
          apply_context->requested_count = user_config.requested_count;
          apply_context->type = user_config.type;
          apply_context->total_frequency++;
          cluster_id_to_apply_contexts_map.insert({cluster_id, apply_context});
        }

        node_to_apply_context_map[node] = cluster_id_to_apply_contexts_map.at(cluster_id);

        // If different plans for the same node have same cluster id, we only need to finalize the first one.
        // The rest of them will be ignored.
        break;
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime::optimizer::memory_optimizer
