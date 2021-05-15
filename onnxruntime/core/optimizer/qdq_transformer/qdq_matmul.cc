// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
class QDQMatMulTransformer : public QDQOperatorTransformer {
 public:
  QDQMatMulTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

  bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) override {
    if (q_nodes.size() == 1) {
      return FuseQLinearMatMul(dq_nodes, q_nodes);
    }
    if (q_nodes.size() == 0) {
      return FuseMatMulIntegerToFloat(dq_nodes);
    }
    return false;
  }

  bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const override {
    constexpr size_t DQCount = 2;
    if (DQCount != dq_nodes.size()) {
      return false;
    }

    if (q_nodes.size() > 0 &&
        (node_.MutableOutputDefs().size() != q_nodes.size() ||
         !optimizer_utils::CheckOutputEdges(graph_, node_, q_nodes.size()))) {
      return false;
    }

    // Currently Quant MatMul only support activation type uint8_t
    int32_t dt = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8;
  }

 private:
  bool FuseQLinearMatMul(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) {
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

  bool FuseMatMulIntegerToFloat(const std::vector<const Node*>& dq_nodes) {
    std::vector<NodeArg*>& input_defs_dq_0 = graph_.GetNode(dq_nodes[0]->Index())->MutableInputDefs();
    std::vector<NodeArg*>& input_defs_dq_1 = graph_.GetNode(dq_nodes[1]->Index())->MutableInputDefs();
    std::vector<NodeArg*> input_defs{
        input_defs_dq_0[0],
        input_defs_dq_1[0],
        input_defs_dq_0[1],
        input_defs_dq_1[1],
        input_defs_dq_0[2],
        input_defs_dq_1[2],
    };

    graph_.AddNode(node_.Name(),
                   "MatMulIntegerToFloat",
                   node_.Description(),
                   input_defs,
                   node_.MutableOutputDefs(),
                   &node_.GetAttributes(),
                   kMSDomain)
        .SetExecutionProviderType(kCpuExecutionProvider);
    return true;
  }
};

DEFINE_QDQ_CREATOR(MatMul, QDQMatMulTransformer)
}  // namespace onnxruntime
