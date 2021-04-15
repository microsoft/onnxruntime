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

 protected:
  bool TransformImpl(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
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

class QDQReshapeTransformer : public QDQSimpleTransformer {
 public:
  QDQReshapeTransformer(Node& node, Graph& graph) : QDQSimpleTransformer(node, graph) {}

 protected:
  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    if (1 != dq_nodes.size() ||  // check that input *data* is output of DequantizeLinear
        node_.MutableOutputDefs().size() != q_nodes.size() ||
        graph_.GetNodeOutputsInGraphOutputs(node_).size() > 0) {
      return false;
    }

    return true;
  }
};

DEFINE_QDQ_CREATOR(MaxPool, QDQSimpleTransformer)
DEFINE_QDQ_CREATOR(Reshape, QDQReshapeTransformer)
}  // namespace onnxruntime
