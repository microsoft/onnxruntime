// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

namespace onnxruntime {
class QDQAveragePoolTransformer : public QDQOperatorTransformer {
 public:
  QDQAveragePoolTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) override {
    std::vector<NodeArg*> input_defs(graph_.GetNode(dq_nodes[0]->Index())->MutableInputDefs());

    Node* q = graph_.GetNode(q_nodes[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);

    graph_.AddNode(node_.Name(),
                   "QLinearAveragePool",
                   node_.Description(),
                   input_defs,
                   q->MutableOutputDefs(),
                   &node_.GetAttributes(),
                   kMSDomain)
        .SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }
};

DEFINE_QDQ_CREATOR(AveragePool, QDQAveragePoolTransformer)
}  // namespace onnxruntime
