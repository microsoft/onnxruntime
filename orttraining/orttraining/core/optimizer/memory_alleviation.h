// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>

#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemoryAlleviation

Find and recompute/offload activations for ops like Dropout/Gelu/Tile.
*/

class MemoryAlleviation : public GraphTransformer {
 private:
  using NodeOutputPort = std::pair<const Node*, int>;
  using ActivationUsedMap = InlinedHashMap<std::string, std::pair<bool, bool>>;

  /**
   * @brief Level to control allowed operations during subgraph detecting.
   * Level 0: only allow cheap-to-compute operations.
   * Level 1: allow more expensive operations.
   */
  enum class ProbeLevel {
    Basic = 0,
    Advanced = 1,
  };

  /**
   * @brief Type of memory reduction techniques.
   */
  enum class AlleviationType {
    None = 0,  // Disabled.
    Recompute = 1,
  };

  struct EntryOperatorConfig {
    InlinedVector<int> input_arg_indices;  // input index to iterate further (bottom up)
  };

 public:
  MemoryAlleviation(const std::string& enable_memory_alleviation, const std::string& level)
      : GraphTransformer("MemoryAlleviation") {
    // Parse user defined alleviation configs.
    ORT_ENFORCE(ParseAlleviationConfigFromString(enable_memory_alleviation).IsOK());
    ORT_ENFORCE(ParseAlleviationLevelFromString(level).IsOK());

    if (static_cast<int>(level_) >= static_cast<int>(ProbeLevel::Basic)) {
      entry_op_type_to_input_arg_index_map_.insert({
          // Binary elementwise
          {"Add", EntryOperatorConfig{{0, 1}}},
          {"Div", EntryOperatorConfig{{0, 1}}},
          {"Mul", EntryOperatorConfig{{0, 1}}},
          {"Sub", EntryOperatorConfig{{0, 1}}},
          {"BiasGelu", EntryOperatorConfig{{0, 1}}},

          // Data layout
          {"Unsqueeze", EntryOperatorConfig{{0}}},
          {"Squeeze", EntryOperatorConfig{{0}}},

          // Unary elementwise
          {"Dropout", EntryOperatorConfig{{0}}},
          {"BitmaskDropout", EntryOperatorConfig{{0}}},
          {"Gelu", EntryOperatorConfig{{0}}},
          {"FastGelu", EntryOperatorConfig{{0}}},

          // Tenary elementwise
          {"Where", EntryOperatorConfig{{0, 1, 2}}},

          // Data copy
          {"Tile", EntryOperatorConfig{{0}}},
          {"Cast", EntryOperatorConfig{{0}}},
      });
    }

    if (static_cast<int>(level_) >= static_cast<int>(ProbeLevel::Advanced)) {
      entry_op_type_to_input_arg_index_map_.insert({
          {"MatMul", EntryOperatorConfig{{0, 1}}},
          {"FusedMatMul", EntryOperatorConfig{{0, 1}}},
          {"Softmax", EntryOperatorConfig{{0}}},
          {"BiasSoftmax", EntryOperatorConfig{{0, 1}}},
          {"BiasSoftmaxDropout", EntryOperatorConfig{{0, 1}}},
      });
    }
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status ParseAlleviationConfigFromString(const std::string& enable_memory_alleviation);

  Status ParseAlleviationLevelFromString(const std::string& level);

  /**
   * @brief Prepare info including activation usage, node usage in fw and bw.
   *
   * @param graph Graph to iterate.
   * @param fw_op_output_arg_used_map Collected activation usage mapping.
   *   - key: node arg name
   *   - value: a pair of bool, representing whether the activation is used by forward nodes or by backward nodes.
   * @param is_forward_op_map Collected map of whether node is used by forward nodes or not.
   *   - key: node index in Graph.
   *   - value: a bool, indicating whether the node is a forward pass op.
   * @param found_yield_op Return whether a YieldOp is found.
   * @return Status
   */
  Status PrepareForTransformation(const Graph& graph,
                                  ActivationUsedMap& fw_op_output_arg_used_map,
                                  InlinedHashMap<NodeIndex, bool>& is_forward_op_map,
                                  bool& found_yield_op) const;
  /**
   * @brief Find all stashed activations, e.g. activations used by forward operators and backward operators.
   *
   * @param graph Graph to iterate.
   * @param fw_op_output_arg_used_map Activation usage mapping.
   * @param candidate_output_args_map Candidate activations, which are consumed by both fw and bw ops.
   * @return Status
   */
  Status GetStashedActivationCandidates(
      const Graph& graph,
      const InlinedHashMap<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
      InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map) const;

  /**
   * @brief Find recomputable subgraphs (has at least one nodes, at most MAXIMUM_RECOMPUTE_NODE_COUNT nodes).
   *
   * @param graph Graph to iterate.
   * @param node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
   * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
   * @param nodes_in_topological_order Collected vector of nodes of found subgraph, in the order of the topological sorted.
   * @param oss string stream to output information to users.
   * @param node_type_in_topological_order string stream for the subgraph node optype sequence, used to info users.
   * @return Status
   */
  Status SelectRecomputeSubgraph(const Graph& graph, const Node& node,
                                 const ActivationUsedMap& fw_op_output_arg_used_map,
                                 InlinedVector<const Node*>& nodes_in_topological_order,
                                 std::ostringstream& oss,
                                 std::ostringstream& node_type_in_topological_order) const;

  /**
   * @brief Duplicate node to create a recompute subgraph.
   *
   * @param graph Graph to iterate.
   * @param nodes_in_topological_order Subgraph nodes to recompute.
   * @param recompute_subgraph_output_node The final node of the subgraph.
   * @return Status
   */
  Status CreateRecomputeGraph(Graph& graph,
                              const InlinedVector<const Node*>& nodes_in_topological_order,
                              Node*& recompute_subgraph_output_node) const;

  /**
   * @brief Convert alleviation type to string.
   *
   * @param type Alleviation type.
   * @return std::string
   */
  std::string AlleviationTypeToString(const AlleviationType& type) const;

  /**
   * @brief Summarize transformation details.
   *
   * @param stashed_activation_statistics statistics around stashed activation memory saving.
   * @return void
   */
  void PrintSummary(const InlinedHashMap<std::string, InlinedHashMap<std::string, int>>&
                        stashed_activation_statistics,
                    const InlinedHashMap<std::string, AlleviationType>&
                        subgraph_str_to_alleviation_type,
                    const logging::Logger& logger) const;

  // The op type that are supported.
  InlinedHashMap<std::string, EntryOperatorConfig> entry_op_type_to_input_arg_index_map_;

  InlinedHashMap<std::string, AlleviationType> pattern_subgraph_to_alleviation_type_map_;

  std::string memory_alleviation_config_;
  ProbeLevel level_;
};

}  // namespace onnxruntime
