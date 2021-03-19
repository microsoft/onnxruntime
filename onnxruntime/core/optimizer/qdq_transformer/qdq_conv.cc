// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

namespace onnxruntime {
class QDQConvTransformer : public QDQOperatorTransformer {
 public:
  QDQConvTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool Transform(const std::vector<const Node*>& parents, const std::vector<const Node*>& children) override {
    if (parents.size() < 2 || children.size() != 1) {
      return false;
    }

    FillQDQOptionalZeroPoint(parents);
    FillQDQOptionalZeroPoint(children);

    std::vector<NodeArg*> input_defs(graph_.GetNode(parents[0]->Index())->MutableInputDefs());
    Node* weight = graph_.GetNode(parents[1]->Index());
    input_defs.insert(input_defs.end(), weight->MutableInputDefs().begin(), weight->MutableInputDefs().end());

    Node* q = graph_.GetNode(children[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);
    if (parents.size() == 3) {
      input_defs.push_back(graph_.GetNode(parents[2]->Index())->MutableInputDefs()[0]);
    }

    Node& qlinear_conv_node = graph_.AddNode(node_.Name(),
                                             "QLinearConv",
                                             node_.Description(),
                                             input_defs,
                                             q->MutableOutputDefs(),
                                             &node_.GetAttributes(),
                                             kOnnxDomain);
    qlinear_conv_node.SetExecutionProviderType(kCpuExecutionProvider);

    return true;
  }
};

DEFINE_QDQ_CREATOR(Conv, QDQConvTransformer)

}  // namespace onnxruntime
