// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

namespace onnxruntime {
class QDQBinaryOpTransformer : public QDQOperatorTransformer {
 public:
  QDQBinaryOpTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    if (children.size() != 1 || parents.size() != 2) {
      return false;
    }

    FillQDQOptionalZeroPoint(parents);
    FillQDQOptionalZeroPoint(children);

    std::vector<NodeArg*> input_defs(graph_.GetNode(parents[0]->Index())->MutableInputDefs());
    Node* b = graph_.GetNode(parents[1]->Index());
    input_defs.insert(input_defs.end(), b->MutableInputDefs().begin(), b->MutableInputDefs().end());

    Node* q = graph_.GetNode(children[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);

    Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                             "QLinear" + node_.OpType(),
                                             node_.Description(),
                                             input_defs,
                                             q->MutableOutputDefs(),
                                             &node_.GetAttributes(),
                                             kMSDomain);
    qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }
};

DEFINE_QDQ_CREATOR(Add, QDQBinaryOpTransformer)
DEFINE_QDQ_CREATOR(Mul, QDQBinaryOpTransformer)

}  // namespace onnxruntime
