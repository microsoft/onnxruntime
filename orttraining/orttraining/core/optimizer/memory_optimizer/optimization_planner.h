// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"

namespace onnxruntime::optimizer::memory_optimizer {

using NodeToClusterApplyContextMap = InlinedHashMap<const Node*, std::shared_ptr<ClusterApplyContext>>;

class MemoryOptimizationPlanner {
 public:
  void AddNodeOptimizationPlan(const Node* node,
                               std::shared_ptr<NodeOptimizationPlanBase> plan) {
    if (node_to_optimization_plans_map.find(node) == node_to_optimization_plans_map.end()) {
      node_to_optimization_plans_map.insert({node, {}});
    }

    node_to_optimization_plans_map[node].emplace_back(plan);
  }

  Status UpdateNodePlansFromExecutionPlan(const Graph& graph,
                                          const OrtValueNameIdxMap& ortvalue_name_to_idx_map,
                                          const SequentialExecutionPlan& p_seq_exec_plan);

  Status FinalizeNodePlansFromUserConfig(
      const InlinedHashMap<std::string, UserConfig>& cluster_id_to_user_configs,
      InlinedHashMap<const Node*, std::shared_ptr<NodeOptimizationPlanBase>>& node_to_opt_plan_map,
      NodeToClusterApplyContextMap& node_to_apply_context_map) const;

  std::string GenerateNodeClusterId(const Node* node) const {
    ORT_ENFORCE(node_to_optimization_plans_map.find(node) != node_to_optimization_plans_map.end(),
                "Node not found in node_to_optimization_plans_map.");
    std::ostringstream oss;
    const auto& node_plans = node_to_optimization_plans_map.at(node);
    for (auto& plan : node_plans) {
      oss << plan->NormalizeForNodeClusterId();
    }

    return oss.str();
  }

  const InlinedHashMap<const Node*,
                       InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>>&
  GetNodeToOptimizationPlanMap() const {
    return node_to_optimization_plans_map;
  }

 private:
  InlinedHashMap<const Node*, InlinedVector<std::shared_ptr<NodeOptimizationPlanBase>>> node_to_optimization_plans_map;
};

}  // namespace onnxruntime::optimizer::memory_optimizer
