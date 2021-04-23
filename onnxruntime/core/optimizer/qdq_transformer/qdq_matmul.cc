// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

namespace onnxruntime {
class QDQMatMulTransformer : public QDQOperatorTransformer {
 public:
  QDQMatMulTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) override {
    std::vector<NodeArg*> input_defs(graph_.GetNode(dq_nodes[0]->Index())->MutableInputDefs());
    Node* b = graph_.GetNode(dq_nodes[1]->Index());
    input_defs.insert(input_defs.end(), b->MutableInputDefs().begin(), b->MutableInputDefs().end());

    Node* q = graph_.GetNode(q_nodes[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);

    graph_.AddNode(node_.Name(),
                   "QLinearMatMul",
                   node_.Description(),
                   input_defs,
                   q->MutableOutputDefs(),
                   &node_.GetAttributes(),
                   kOnnxDomain)
        .SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }

  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    if (!QDQOperatorTransformer::Check(dq_nodes, q_nodes)) {
      return false;
    }

    // Currently QLinearConv only support activation type uint8_t
    int32_t dt = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8;
  }
};

DEFINE_QDQ_CREATOR(MatMul, QDQMatMulTransformer)
}  // namespace onnxruntime
