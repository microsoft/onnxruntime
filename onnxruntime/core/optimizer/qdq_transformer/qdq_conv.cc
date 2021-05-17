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

 protected:
  bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) override {
    std::vector<NodeArg*> input_defs(graph_.GetNode(dq_nodes[0]->Index())->MutableInputDefs());
    Node* weight = graph_.GetNode(dq_nodes[1]->Index());
    input_defs.insert(input_defs.end(), weight->MutableInputDefs().begin(), weight->MutableInputDefs().end());

    Node* q = graph_.GetNode(q_nodes[0]->Index());
    input_defs.push_back(q->MutableInputDefs()[1]);
    input_defs.push_back(q->MutableInputDefs()[2]);
    if (dq_nodes.size() == 3) {
      input_defs.push_back(graph_.GetNode(dq_nodes[2]->Index())->MutableInputDefs()[0]);
    }

    graph_.AddNode(node_.Name(),
                   "QLinearConv",
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

    // Currently QLinearConv only support activation type uint8_t and output type uint8_t
    int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_input != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
        dt_output != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
      return false;
    }

    if (dq_nodes.size() < 3) {  // no bias
      return true;
    }

    int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
  }
};

DEFINE_QDQ_CREATOR(Conv, QDQConvTransformer)

}  // namespace onnxruntime
