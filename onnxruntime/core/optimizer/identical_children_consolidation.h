// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

using ONNX_NAMESPACE::TensorProto;
using std::string_view;
using std::unordered_map;
using std::unordered_set;

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
 *             [supported_parent_ops]
 *                /              \
 * [supported_children_ops] [supported_children_ops]
 *              |                 |
 *      [grandchildren_a]   [grandchildren_b]
 *
 * will be transformed to:
 *
 *            [supported_parent_ops]
 *                      |
 *           [supported_children_ops]
 *               /              \
 *     [grandchildren_a]   [grandchildren_b]
 */
class IdenticalChildrenConsolidation : public GraphTransformer {
 public:
  IdenticalChildrenConsolidation() : GraphTransformer("IdenticalChildrenConsolidation") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool IsSupportedParentNode(const Node* node) const;
  std::vector<std::vector<NodeIndex> > DivideIdenticalChildrenIntoGroups(const Graph& graph, Node* node, const string_view& op) const;
  string_view IdentityBuilder(const Graph& graph, const Node& node) const;

  unordered_map<string_view, unordered_set<string_view> > supported_ops = {
      {"DequantizeLinear", {"QuantizeLinear"}},
      {"QuantizeLinear", {"DequantizeLinear"}}};
  string_view constant_prefix = "ItIsSpecialConstantPrefix_";
  string_view ignore_identity = "IgNoReD_IdEnTiTy";
};
}  // namespace onnxruntime
