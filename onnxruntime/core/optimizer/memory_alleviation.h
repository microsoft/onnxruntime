// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemoryAlleviation

Set priority for recomputed nodes for example: Gelu/BiasGelu/FastGelu.

*/
class MemoryAlleviation : public GraphTransformer {
 public:
  MemoryAlleviation(
      const int32_t enable_gelu_recompute,
      const int32_t enable_dropout_recompute,
      const int32_t enable_tile_recompute) noexcept
      : GraphTransformer("MemoryAlleviation"),
        enable_gelu_recompute_{enable_gelu_recompute},
        enable_dropout_recompute_{enable_dropout_recompute},
        enable_tile_recompute_{enable_tile_recompute} {
    cheap_to_recompute_op_type_list_["Where"] =
        [](const Graph& graph,
           const Node& node,
           std::vector<std::pair<const Node*, int>> next_input_args) -> bool {
      const Node* data_true_node = graph.GetProducerNode(node.InputDefs()[1]->Name());
      size_t producer_output_index = 0;
      for (size_t i = 0; i < data_true_node->OutputDefs().size(); ++i) {
        if (data_true_node->OutputDefs()[i]->Name().compare(node.InputDefs()[1]->Name()) == 0) {
          producer_output_index = i;
          break;
        }
      }

      const ONNX_NAMESPACE::TensorProto* false_initializer = graph.GetConstantInitializer(node.InputDefs()[2]->Name(),
                                                                                          true);
      if (!false_initializer) {
        return false;
      }

      next_input_args.push_back(std::make_pair<const Node*, int>(std::move(data_true_node), producer_output_index));
      return true;
    };

    if (enable_gelu_recompute_) {
      recompute_op_type_to_input_arg_index_map_.insert({{"Gelu", {0}}, {"FastGelu", {0}}, {"BiasGelu", {0}}});
    }

    if (enable_dropout_recompute_) {
      recompute_op_type_to_input_arg_index_map_.insert({{"Dropout", {0}}, {"BitmaskDropout", {0}}});
    }

    if (enable_tile_recompute_) {
      recompute_op_type_to_input_arg_index_map_.insert({{"Tile", {0}}});
    }
  }

  Status
  ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status PrepareCandidateNodes(
      Graph& graph,
      std::unordered_map<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
      std::unordered_map<NodeIndex, bool>& is_forward_op_map) const;

  Status CheckRecomputeCondition(
      Graph& graph, const Node& node,
      std::vector<const Node*>& nodes,
      const std::unordered_map<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map) const;

  int32_t enable_gelu_recompute_;
  int32_t enable_dropout_recompute_;
  int32_t enable_tile_recompute_;

  std::unordered_map<std::string, std::function<bool(const Graph& graph, const Node&, std::vector<std::pair<const Node*, int>> next_input_args)>> cheap_to_recompute_op_type_list_;
  std::unordered_map<std::string, std::vector<int>> recompute_op_type_to_input_arg_index_map_;
};

}  // namespace onnxruntime
