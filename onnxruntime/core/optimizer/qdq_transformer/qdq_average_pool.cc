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

 protected:
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

  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    if (!QDQOperatorTransformer::Check(dq_nodes, q_nodes)) {
      return false;
    }

    // Currently QLinearAveragePool only support activation type uint8_t
    int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 &&
           dt_output == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8;
  }
};

DEFINE_QDQ_CREATOR(AveragePool, QDQAveragePoolTransformer)
}  // namespace onnxruntime
