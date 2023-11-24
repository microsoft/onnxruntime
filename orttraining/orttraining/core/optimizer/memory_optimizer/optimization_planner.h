// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"

namespace onnxruntime::optimizer::memory_optimizer {

/**
 * @brief Struct to store properties of a specific subgraph.
 */
class ClusterApplyContext {
 public:
  ClusterApplyContext() = default;

  OptimizationType type;
  int requested_count{0};
  int total_frequency{0};  // The occurrence of this subgraph pattern in the graph.

  int applied_count{0};  // The number of times this subgraph pattern has been really applied in this transformer.
  int skip_count{0};     // The number of times this subgraph instance has been skipped in reversed topological order.
};

/**
 * @brief Base class for a concrete optimization plan.
 *
 */
class NodeOptimizationPlanBase {
 public:
  NodeOptimizationPlanBase(const Node* node,
                           gsl::span<const size_t> activation_output_indices,
                           float save_ratio)
      : node(node),
        activation_output_indices_(activation_output_indices.begin(), activation_output_indices.end()),
        save_ratio_(save_ratio) {
  }

  virtual ~NodeOptimizationPlanBase() = default;

  virtual OptimizationType GetOptimizationType() const = 0;

  /**
   * Get the cluster id for this optimization plan.
   * This cluster id is used to enable the optimization as a unique identity, for example, for recompute it is a
   * subgraph string representation.
   * @return std::string
   */
  virtual std::string GetClusterId() const = 0;

  /**
   * Get a string used to generate node cluster id for this optimization plan.
   * Node cluster id is on Node level, each node can have multiple optimization plans, each plan generates its
   * normalization string. Once combined we get Node cluster id. This id is used to categorize nodes into different
   * groups, showing them as one row in memory optimization opportunity table.
   * @return std::string
   */
  virtual std::string NormalizeForNodeClusterId() const = 0;

  /**
   * Return all output indices that are used as activation buffers.
   */
  gsl::span<const size_t> GetActivationOutputIndices() const { return activation_output_indices_; }

  /**
   * Return the saving ratio for this optimization plan.
   */
  float GetSaveRatio() const { return save_ratio_; }

  /**
   * Get a symbolic string to represent the memory saving for this optimization plan.
   */
  std::string GetMemorySavingSymbolicString() const;

  const Node* node;
  // A map: output index reusing other node's output (other_node, output index)
  InlinedHashMap<size_t, NodeOutputPort> reuse_buffers;

 private:
  InlinedVector<size_t> activation_output_indices_;
  float save_ratio_ = 1.0f;
};

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

  Status UpdateNodePlansFromExecutionPlan(const GraphViewer& graph_viewer,
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
