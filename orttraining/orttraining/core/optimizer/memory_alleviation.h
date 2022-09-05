// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <charconv>

#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

namespace memory_alleviation {

/**
 * @brief Type of memory reduction techniques.
 */
enum AlleviationType {
  None = 0,  // Disabled.
  Recompute = 1,
};

constexpr int32_t MAXIMUM_RECOMPUTE_COUNT = 3;
constexpr char UserConfig_OpTypeDropout[] = "Dropout";
constexpr char UserConfig_OpTypeGelu[] = "Gelu";
constexpr char UserConfig_OpTypeTile[] = "Tile";

using NodeOutputPort = std::pair<const Node*, int>;
using OpCrawlerFunctionType = std::function<bool(const Graph&, const Node&, InlinedVector<NodeOutputPort>&)>;
using ActivationUsedMap = InlinedHashMap<std::string, std::pair<bool, bool>>;

struct EntryOperatorConfig {
  InlinedVector<int> input_arg_indices;  // input index to iterate further (bottom up)
  AlleviationType type;
};

}  // namespace memory_alleviation

/**
@Class MemoryAlleviation

Find and recompute/offload activations for ops like Dropout/Gelu/Tile.
*/

class MemoryAlleviation : public GraphTransformer {
 public:
  MemoryAlleviation(const std::string& enable_memory_alleviation) : GraphTransformer("MemoryAlleviation") {
    // Parse user defined alleviation configs.
    ORT_ENFORCE(ParseAlleviationConfigFromString(enable_memory_alleviation).IsOK());
    entry_op_type_to_input_arg_index_map_.insert(
        {{"Gelu", memory_alleviation::EntryOperatorConfig{{0}, gelu_alleviation_type_}},
         {"FastGelu", memory_alleviation::EntryOperatorConfig{{0}, gelu_alleviation_type_}},
         {"BiasGelu", memory_alleviation::EntryOperatorConfig{{0}, gelu_alleviation_type_}}});

    entry_op_type_to_input_arg_index_map_.insert(
        {{"Dropout", memory_alleviation::EntryOperatorConfig{{0}, dropout_alleviation_type_}},
         {"BitmaskDropout", memory_alleviation::EntryOperatorConfig{{0}, dropout_alleviation_type_}}});

    entry_op_type_to_input_arg_index_map_.insert(
        {{"Tile", memory_alleviation::EntryOperatorConfig{{0}, tile_alleviation_type_}}});

    ORT_ENFORCE(RegisterRecomputableIntermediateOps().IsOK());
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status ParseAlleviationConfigFromString(const std::string& enable_memory_alleviation);

  Status RegisterRecomputableIntermediateOps();

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
   * @return Status
   */
  Status PrepareForTransformation(const Graph& graph,
                                  memory_alleviation::ActivationUsedMap& fw_op_output_arg_used_map,
                                  InlinedHashMap<NodeIndex, bool>& is_forward_op_map) const;
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
   * @brief Find recomputable subgraphs (has at least one nodes, at most MAXIMUM_RECOMPUTE_COUNT nodes).
   *
   * @param graph Graph to iterate.
   * @param node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
   * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
   * @param nodes_in_topological_order Collected vector of nodes of found subgraph, in the order of the topological sorted.
   * @param oss string stream to output information to users.
   * @param node_type_in_topological_order string stream for the subgraph node optype sequence, used to info users.
   * @param alleviation_type return the alleviation type of the passed-in node.
   * @return Status
   */
  Status SelectSubgraph(const Graph& graph, const Node& node,
                        const memory_alleviation::ActivationUsedMap& fw_op_output_arg_used_map,
                        InlinedVector<const Node*>& nodes_in_topological_order,
                        std::ostringstream& oss,
                        std::ostringstream& node_type_in_topological_order,
                        memory_alleviation::AlleviationType& alleviation_type) const;

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
   * @brief Summarize transformation details.
   *
   * @param stashed_activation_statistics statistics around stashed activation memory saving.
   * @return void
   */
  void PrintSummary(const InlinedHashMap<std::string, InlinedHashMap<std::string, int>>&
                        stashed_activation_statistics,
                    const InlinedHashMap<std::string, memory_alleviation::AlleviationType>&
                        subgraph_str_to_alleviation_type) const;

  // For some computational cheap operator, register whether/how to extend recompute subgraph into its input nodes.
  InlinedHashMap<std::string, memory_alleviation::OpCrawlerFunctionType> recomputable_intermediate_op_crawler_map_;

  // The op type that used as an ending node (or entry node) to find a recompute subgraph or find an offload activation.
  InlinedHashMap<std::string, memory_alleviation::EntryOperatorConfig> entry_op_type_to_input_arg_index_map_;

  memory_alleviation::AlleviationType gelu_alleviation_type_{memory_alleviation::AlleviationType::None};
  memory_alleviation::AlleviationType dropout_alleviation_type_{memory_alleviation::AlleviationType::None};
  memory_alleviation::AlleviationType tile_alleviation_type_{memory_alleviation::AlleviationType::None};
};

}  // namespace onnxruntime
