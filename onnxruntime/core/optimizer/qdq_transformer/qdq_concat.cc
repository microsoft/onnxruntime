// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"
#include "core/optimizer/qdq_transformer/registry.h"

#include <vector>

namespace onnxruntime {

class QDQConcatTransformer : public QDQOperatorTransformer {
 public:
  QDQConcatTransformer(Node& node, Graph& graph) : QDQOperatorTransformer(node, graph) {}

 protected:
  bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) override {
    std::vector<NodeArg*> fp32_input_defs = node_.MutableInputDefs();
    auto input_count = fp32_input_defs.size();
    if (input_count != dq_nodes.size() || q_nodes.size() != 1) {
      return false;
    }

    std::vector<NodeArg*> input_defs;
    input_defs.reserve(2 + input_count * 3);

    Node* output_qnode = graph_.GetNode(q_nodes[0]->Index());
    input_defs.push_back(output_qnode->MutableInputDefs()[1]);
    input_defs.push_back(output_qnode->MutableInputDefs()[2]);

    for (size_t input_index = 0; input_index < input_count; ++input_index) {
        auto qinput_defs = graph_.GetNode(dq_nodes[input_index]->Index())->MutableInputDefs();
        input_defs.insert(input_defs.end(), qinput_defs.begin(), qinput_defs.end());
    }

    graph_.AddNode(node_.Name(),
                   "QLinearConcat",
                   node_.Description(),
                   input_defs,
                   output_qnode->MutableOutputDefs(),
                   &node_.GetAttributes(),
                   kOnnxDomain)
        .SetExecutionProviderType(kCpuExecutionProvider);

    return true;
  }
};

DEFINE_QDQ_CREATOR(Concat, QDQConcatTransformer)

}  // namespace onnxruntime
