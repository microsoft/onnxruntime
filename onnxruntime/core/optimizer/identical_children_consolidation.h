// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

using std::unordered_map;
using std::unordered_set;
using std::string_view;

/**
 * @Class IdenticalChildrenConsolidation
 *
 * This transformer consolidates identical children nodes in a graph. The consolidate children
 * Must have the same parent and have edges with same attributes expect different destination node.
 * Currently, it only supports nodes with single input and single output and the following node
 * types from supported_ops list and supported_children_ops list.
 *
 * For example, the following graph
 *
 *                [supported_ops]
 *                /              \
 * [supported_children_ops] [supported_children_ops]
 *              |                 |
 *          [any_ops_a]      [any_ops_b]
 *
 * will be transformed to:
 *
 *                [supported_ops]
 *                      |
 *           [supported_children_ops]
 *               /              \
 *        [any_ops_a]     [any_ops_b]
 */
class IdenticalChildrenConsolidation : public GraphTransformer {
 public:
  IdenticalChildrenConsolidation() : GraphTransformer("IdenticalChildrenConsolidation") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  unordered_set<string_view> supported_children_optypes = {"DequantizeLinear","QuantizeLinear"} ;
  unordered_set<string_view> supported_parent_optypes = {"DequantizeLinear","QuantizeLinear"} ;
  std::vector<NodeIndex> GetIndenticalChildrenSet (Node& node) const;
};
}  // namespace onnxruntime



