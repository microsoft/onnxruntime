// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"

namespace onnxruntime::optimizer::memory_optimizer {

/**
 * @brief Level to control allowed operations during subgraph detecting.
 * Level 0: only allow cheap-to-compute operations.
 * Level 1: allow more expensive operations.
 */
enum class ProbeLevel {
  Basic = 0,
  Advanced = 1,
  LevelMax = 2,
};

/**
 * @brief Configuration to control recompute subgraph detection.
 */
class ProbeConfig {
 public:
  ProbeConfig() = default;

  ProbeConfig(ProbeLevel level, bool transformer_layer_as_boundary = false) {
    probe_level = level;
    enable_transformer_layer_as_boundary = transformer_layer_as_boundary;
  }

  ProbeLevel probe_level{ProbeLevel::Basic};
  bool enable_transformer_layer_as_boundary{false};
};

Status ParseProbeConfigFromString(std::string_view recompute_probe_config,
                                  ProbeConfig& probe_config);

/**
 * @brief A child class used for Recompute/RecomputeWithCompromise optimization plan.
 *
 * For each node generating stashed activations, a recompute plan can be created for it.
 */
class NodeRecomputePlan : public NodeOptimizationPlanBase {
 public:
  NodeRecomputePlan(const Node* node,
                    const InlinedVector<size_t>& activation_output_indices,
                    const InlinedVector<const Node*>& nodes_in_topological_order,
                    bool compromise_recompute = false,
                    float save_ratio = 1.0f) : NodeOptimizationPlanBase(node, activation_output_indices, save_ratio) {
    compromise_recompute_ = compromise_recompute;
    // Be noted, recompute is node level, each node arg should have the same optimization type.
    nodes_in_topological_order_ = nodes_in_topological_order;
  }

  const InlinedVector<const Node*>& GetNodesInTopoOrder() const { return nodes_in_topological_order_; }

  bool IsCompromiseRecompute() const { return compromise_recompute_; }

  OptimizationType GetOptimizationType() const override {
    return compromise_recompute_ ? OptimizationType::RecomputeWithCompromise
                                 : OptimizationType::Recompute;
  }

  /**
   * @brief Get the cluster id for this recompute plan.
   * The cluster id is used to identify a unique subgraph.
   * User can pass such cluster id to enable specific memory optimization for some subgraph.
   */
  std::string GetClusterId() const override;

  /**
   * @brief Get the serialized string for this recompute plan to create Node-level cluster id.
   * Imagine, a Node can have multiple optimization plans, each plan generates its normalization string.
   * Once combined we get Node cluster id.
   *
   * Node cluster id is used to categorize nodes into different groups, showing them as one row in memory
   * optimization opportunity table.
   */
  std::string NormalizeForNodeClusterId() const override;

  std::string GetNodesInTopoOrderStr() const;

  std::string GetMemorySavingSymbolicString() const override {
    std::string saving_str;
    for (auto output_index : GetActivationOutputIndices()) {
      // If the output is reusing other node's buffer, then no memory saving.
      std::string cur_output_saving_str;

      bool is_reused = reuse_buffers.find(output_index) != reuse_buffers.end();
      bool is_src_node_in_cur_node_subgraph = false;
      if (is_reused) {
        // Here we assume the src_node is the real owner of the buffer, so we don't need trace further.
        const auto* src_node = reuse_buffers.at(output_index).first;
        is_src_node_in_cur_node_subgraph = std::find(nodes_in_topological_order_.begin(),
                                                     nodes_in_topological_order_.end(),
                                                     src_node) != nodes_in_topological_order_.end();
      }

      if (!is_reused || is_src_node_in_cur_node_subgraph) {
        // For is_src_node_in_cur_node_subgraph is True, still use the output to calculate the saving, because
        // reusing buffer is the same size.
        const auto& output_def = node->OutputDefs()[output_index];
        MLDataType ml_data_type = DataTypeImpl::TypeFromProto(*output_def->TypeAsProto());
        ORT_ENFORCE(ml_data_type->IsTensorType(), "ml_type must be a tensor type, but it is ",
                    DataTypeImpl::ToString(ml_data_type));
        const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
        ORT_ENFORCE(nullptr != tensor_type_base);
        MLDataType elt_type = tensor_type_base->GetElementType();
        const auto byte_count_per_element = elt_type->Size();
        cur_output_saving_str = GetActivationOutputDimParamString(output_index) + " * " +
                                std::to_string(byte_count_per_element) + " * " +
                                std::to_string(GetSaveRatio());
      } else {
        cur_output_saving_str = "0";
      }

      if (!saving_str.empty()) {
        saving_str += " + ";
      }

      saving_str = "(" + cur_output_saving_str + ")";
    }

    ORT_ENFORCE(!saving_str.empty(), "saving_str should not be empty for node: ", node->OpType(), " ", node->Name());
    return "(" + saving_str + ")";
  }

 private:
  bool compromise_recompute_;
  InlinedVector<const Node*> nodes_in_topological_order_;
};

/**
 * @brief For the node producing stashed activation, check whether a recomputable subgraph can be found or not.
 *
 * @param graph_viewer The graph viewer to get node information.
 * @param node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
 * @param probe_config The config for subgraph detecting.
 * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
 * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
 *   Used to re-order the collected subgraph nodes.
 * @param candidate_output_args_map A map from node to its candidate activations, which are consumed by both fw and
 *  bw ops.
 * @param layer_boundary_ln_nodes A set of LayerNormalization nodes, which are used as the boundary for subgraph.
 * @param subgraph_stores A store to maintain all found subgraphs.
 * @param logger Logger.
 * @param compromise_stashed_activation Whether to compromise stashed activation, e.g. if we cannot find a
 * recomputable subgraph to save a stashed activation, we can compromise to find a recomputable subgraph to reduce the
 * size of stashed activation.
 * @param can_compromise_stashed_activation A bool return value, to indicate there is opportunaties for finding a
 * compromised subgraph.
 */
std::unique_ptr<NodeRecomputePlan> CheckNodeForRecompute(const GraphViewer& graph_viewer,
                                                         const Node& node,
                                                         const ProbeConfig& probe_config,
                                                         const ActivationUsedMap& fw_op_output_arg_used_map,
                                                         const InlinedHashMap<NodeIndex, ptrdiff_t>&
                                                             node_index_to_its_order_in_topological_sort_map,
                                                         const InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                                             candidate_output_args_map,
                                                         const InlinedVector<const Node*>& layer_boundary_ln_nodes,
                                                         const logging::Logger& logger,
                                                         bool compromise_stashed_activation,
                                                         bool& can_compromise_stashed_activation);

}  // namespace onnxruntime::optimizer::memory_optimizer
