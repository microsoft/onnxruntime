// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

namespace onnxruntime {
class QDQSimpleTransformer : public QDQOperatorTransformer {
 public:
  QDQSimpleTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    if (parents.size() != 1 || children.size() != 1) {
      return false;
    }

    FillQDQOptionalZeroPoint(parents);
    FillQDQOptionalZeroPoint(children);

    graph_.RemoveEdge(parents[0]->Index(), node_.Index(), 0, 0);
    graph_.RemoveEdge(node_.Index(), children[0]->Index(), 0, 0);

    node_.MutableInputDefs()[0] = graph_.GetNode(parents[0]->Index())->MutableInputDefs()[0];
    node_.MutableOutputDefs()[0] = graph_.GetNode(children[0]->Index())->MutableOutputDefs()[0];
    return true;
  }

  bool KeepNode() const override {
    return true;
  }
};

DEFINE_QDQ_CREATOR(MaxPool, QDQSimpleTransformer)
DEFINE_QDQ_CREATOR(Reshape, QDQSimpleTransformer)
}  // namespace onnxruntime
