// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/registry.h"
#include "core/optimizer/utils.h"

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

  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    if (1 != dq_nodes.size() ||  // check that input *data* is output of DequantizeLinear
        1 != q_nodes.size() ||
        !optimizer_utils::CheckOutputEdges(graph_, node_, 1)) {
      return false;
    }

    return QDQ::IsQDQPairSupported(graph_, *q_nodes[0], *dq_nodes[0]);
  }
};

class QDQMaxPoolTransformer : public QDQSimpleTransformer {
 public:
  QDQMaxPoolTransformer(Node& node, Graph& graph) : QDQSimpleTransformer(node, graph) {}

 protected:
  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    if (!QDQSimpleTransformer::Check(dq_nodes, q_nodes)) {
      return false;
    }

    return graph_utils::IsSupportedOptypeVersionAndDomain(node_, "MaxPool", {12});
  }
};

DEFINE_QDQ_CREATOR(MaxPool, QDQMaxPoolTransformer)
DEFINE_QDQ_CREATOR(Reshape, QDQSimpleTransformer)
DEFINE_QDQ_CREATOR(Gather, QDQSimpleTransformer)
DEFINE_QDQ_CREATOR(Transpose, QDQSimpleTransformer)
}  // namespace onnxruntime
