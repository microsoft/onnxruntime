// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <charconv>
#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemoryOptimizer

Find recomputable subgraphs and enable according to user configs.
*/

class MemoryOptimizer : public GraphTransformer {
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
    LevelMax = 2,
  };

  /**
   * @brief Type of memory reduction techniques.
   */
  enum class OptimizationType {
    None = 0,  // Disabled.
    Recompute = 1,
    TypeMax = 2,
  };

  /**
   * @brief Type of user config.
   * type: type of memory reduction techniques.
   * requested_count: the number of occurrences of a subgraph pattern for alleviation. -1 means apply all.
   *   One example: if a subgraph pattern is found 3 times, and requested_count is set 2, then the 1st and 2nd subgraph
   *   in topological order will be applied for alleviation. This is useful to avoid alleviating more memory than
   *   needed.
   */
  struct UserConfig {
    OptimizationType type;
    int requested_count;
  };

  /**
   * @brief Struct to store properties of a specific subgraph.
   */
  struct SubGraphDesc {
    SubGraphDesc() = default;

    // A string to represent the subgraph, used as a unique "ID" for a unique subgraph.
    std::string subgraph_representative_str;

    InlinedHashMap<std::string, int> shape_str_frequency;  // shape string to frequency
    UserConfig user_optimizer_config;
    int total_frequency{0};  // The occurrence of this subgraph pattern in the graph.

    int applied_count{0};      // The number of times this subgraph pattern has been really applied in this transformer.
    int skip_count{0};         // The number of times this subgraph instance has been skipped in reversed topological order.
    float saving_ratio{1.0f};  // For compromised memory saving, the ratio of memory saving.
  };

  /**
   * @brief A struct to maintain the information of target subgraphs to optimize.
   * Imagine we loop all nodes finding recomputable/offload-able subgraphs, we want to store them first.
   * Afterwards, we optionally pick up some of them to apply optimization according to user configs.
   *
   * subgraph_descs is a map from subgraph string representation to its subgraph related configurations.
   *
   * _optimization_target_graphs_ is a map from activation producer node pointers to its target optimization subgraph
   * nodes. For example, if a subgraph Cast+Gelu can be recomputed, we may have a map like:
   *  key: node pointer of stashed activation producer Gelu; value: node vector {Cast, Gelu,}.
   *
   * When we AddSubGraphInstance, we must provider its corresponding subgraph desc in the parameter.
   * Then we can know for each subgraph instance, what's the subgraph str representation, and what's the optimization
   * config.
   */
  struct SubGraphStores {
    /**********************************
    ** subgraph desc section starts **
    **********************************/

    size_t SubGraphDescCount() const {
      return subgraph_descs.size();
    }

    bool Contains(std::string_view subgraph_str) const {
      return subgraph_descs.find(subgraph_str) != subgraph_descs.end();
    }

    SubGraphDesc& GetSubGraphDesc(std::string_view subgraph_string) {
      ORT_ENFORCE(Contains(subgraph_string), "Subgraph string not found.", subgraph_string);
      return subgraph_descs.at(subgraph_string);
    }

    SubGraphDesc& CreateSubGraphDesc(const std::string& subgraph_string,
                                     UserConfig& config) {
      ORT_ENFORCE(!Contains(subgraph_string), "Subgraph string already exists.", subgraph_string);
      subgraph_descs[subgraph_string].user_optimizer_config = config;
      subgraph_descs[subgraph_string].subgraph_representative_str = subgraph_string;
      return subgraph_descs[subgraph_string];
    }

    /**********************************************************************
    ** subgraph desc section ends, and subgraph instance section starts. **
    ***********************************************************************/

    // Pair of <nodes in topological order, a string to represent the subgraph>.
    using GraphInstanceInfo = std::pair<InlinedVector<const Node*>, std::string>;

    void AddSubGraphInstance(const Node* node,
                             const InlinedVector<const Node*>& nodes_in_topological_order,
                             const SubGraphDesc& subgraph_desc) {
      ORT_ENFORCE(_optimization_target_graphs_.find(node) == _optimization_target_graphs_.end());
      _optimization_target_graphs_[node] = std::make_pair(nodes_in_topological_order,
                                                          subgraph_desc.subgraph_representative_str);
    }

    bool ContainsSubGraphInstance(const Node* node) const {
      return _optimization_target_graphs_.find(node) != _optimization_target_graphs_.end();
    }

    GraphInstanceInfo& GetSubGraphInstance(const Node* node) {
      ORT_ENFORCE(_optimization_target_graphs_.find(node) != _optimization_target_graphs_.end());
      return _optimization_target_graphs_[node];
    }

    /***********************************
    ** subgraph instance section ends **
    ***********************************/

    InlinedHashMap<std::string /*subgraph_representative_str*/, SubGraphDesc> subgraph_descs;
    InlinedHashMap<const Node*, GraphInstanceInfo> _optimization_target_graphs_;
  };

  /**
   * @brief Used to define per-op recompute config.
   *
   */
  struct AllowedRecomputeNodeConfig {
    InlinedVector<int> input_arg_indices;  // input index to iterate further (bottom up)
  };

 public:
  MemoryOptimizer(const std::string& enable_memory_optimizer, const std::string& level)
      : GraphTransformer("MemoryOptimizer") {
    // Parse user defined configs.
    ORT_ENFORCE(ParseConfigFromString(enable_memory_optimizer, level).IsOK());

    RegisterAllowedRecomputeOps();
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status ParseConfigFromString(const std::string& enable_memory_optimizer, const std::string& level);

  /**
   * @brief Prepare info including activation usage, node usage in fw and bw.
   *
   * @param graph Graph to iterate.
   * @param fw_op_output_arg_used_map Collected activation usage mapping.
   *   - key: node arg name
   *   - value: a pair of bool, representing whether the activation is used by forward nodes or by backward nodes.
   * @return int64_t value The boundary op (for example YieldOp) order in topological order. If no boundary op found,
   *  return -1;
   */
  int64_t PrepareForTransformation(const Graph& graph,
                                   ActivationUsedMap& fw_op_output_arg_used_map,
                                   InlinedHashMap<NodeIndex, size_t>&
                                       node_index_to_its_order_in_topological_sort_map) const;
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
      InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map,
      const logging::Logger& logger) const;

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
                   const InlinedHashMap<NodeIndex, size_t>& node_index_to_its_order_in_topological_sort_map,
                   const InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map,
                   const logging::Logger& logger,
                   int64_t boundary_op_order_in_topological_sort,
                   SubGraphStores& subgraph_stores,
                   Node* node) const;

  /**
   * @brief Convert the recompute subgraph to its string representation.
   *
   * @param nodes_in_topological_order The subgraph nodes in topological order.
   * @param subgraph_string_representation Returns subgraph string representation.
   * @param log_info Returns log info for users.
   */
  void NodesInTopoOrderToString(const InlinedVector<const Node*>& nodes_in_topological_order,
                                std::string& subgraph_string_representation,
                                std::string& log_info) const;

  /**
   * @brief Convert optimization type to string.
   */
  std::string UserConfigToString(const UserConfig& config) const;

  /**
   * @brief Summarize transformation details.
   *
   * @param stashed_activation_statistics statistics around stashed activation memory saving.
   * @return void
   */
  void PrintSummary(const SubGraphStores& recompute_stores,
                    const SubGraphStores& recompute_with_compromise_stores,
                    const logging::Logger& logger) const;

  /**************************************************
   ** Recompute related function definition starts **
   *************************************************/

  void RegisterAllowedRecomputeOps();

  /**
   * @brief Find recomputable subgraphs (has at least one nodes, at most MAXIMUM_RECOMPUTE_NODE_COUNT nodes).
   *
   * @param node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
   * @param node_output_index_candidates Candidate output indices of "node", which are consumed by both fw and bw ops.
   * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
   * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
   *   Used to re-order the collected subgraph nodes.
   * @param nodes_in_topological_order Collected vector of nodes of found subgraph, in the order of the topological
   *  sorted.
   * @param logger Logger.
   * @param compromise_stashed_activation Whether to compromise stashed activation, e.g. if we cannot find a
   * recomputable subgraph to save a stashed activation, we can compromise to find a recomputable subgraph to reduce the
   * size of stashed activation.
   * @param can_compromise_stashed_activation A bool return value, to indicate there is opportunaties for finding a
   * compromised subgraph.
   * @return Status
   */
  Status SelectRecomputeSubgraph(const Node& node,
                                 const InlinedVector<size_t>& node_output_index_candidates,
                                 const ActivationUsedMap& fw_op_output_arg_used_map,
                                 const InlinedHashMap<NodeIndex, size_t>&
                                     node_index_to_its_order_in_topological_sort_map,
                                 InlinedVector<const Node*>& nodes_in_topological_order,
                                 const logging::Logger& logger,
                                 bool compromise_stashed_activation,
                                 bool& can_compromise_stashed_activation) const;

  /**
   * @brief For the node producing stashed activation, check whether a recomputable subgraph can be found or not.
   *
   * @param node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
   * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
   * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
   *   Used to re-order the collected subgraph nodes.
   * @param candidate_output_args_map A map from node to its candidate activations, which are consumed by both fw and
   *  bw ops.
   * @param subgraph_stores A store to maintain all found subgraphs.
   * @param logger Logger.
   * @param compromise_stashed_activation Whether to compromise stashed activation, e.g. if we cannot find a
   * recomputable subgraph to save a stashed activation, we can compromise to find a recomputable subgraph to reduce the
   * size of stashed activation.
   * @param can_compromise_stashed_activation A bool return value, to indicate there is opportunaties for finding a
   * compromised subgraph.
   */
  void CheckNodeForRecompute(const Node& node,
                             const ActivationUsedMap& fw_op_output_arg_used_map,
                             const InlinedHashMap<NodeIndex, size_t>&
                                 node_index_to_its_order_in_topological_sort_map,
                             const InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                 candidate_output_args_map,
                             SubGraphStores& subgraph_stores,
                             const logging::Logger& logger,
                             bool compromise_stashed_activation,
                             bool& can_compromise_stashed_activation) const;

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

  // The op types that are supported predefined.
  InlinedHashMap<std::string, AllowedRecomputeNodeConfig> recomputable_op_type_to_input_arg_index_map_;
  // User enabled map of the subgraph string representation to the alleviation type.
  InlinedHashMap<std::string, UserConfig> pattern_subgraph_to_user_optimizer_config_map_;
  std::string optimizer_config_;
  ProbeLevel recompute_probe_level_;
};

}  // namespace onnxruntime
