// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/loss_rewriter.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status SoftmaxCrossEntropyLossInternalFusion::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                                        const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (!p_node || p_node->OpType() != "NegativeLogLikelihoodLossInternal") {
      continue;
    }

    Node& loss_node = *p_node;
    const Node* p_first_input = graph_utils::GetInputNode(loss_node, 0);
    if (!p_first_input || (p_first_input->OpType() != "Cast" && p_first_input->OpType() != "LogSoftmax")) {
      continue;
    }

    if (p_first_input->OpType() == "Cast") {
      p_first_input = graph_utils::GetInputNode(*p_first_input, 0);
      if (!p_first_input || p_first_input->OpType() != "LogSoftmax") {
        continue;
      }
    }

    Node& log_softmax_node = *graph.GetNode(p_first_input->Index());
    graph_utils::RemoveNode(graph, log_softmax_node);
    const auto& loss_inputs = loss_node.MutableInputDefs();
    auto& loss_outputs = loss_node.MutableOutputDefs();
    auto input_def_type_proto = loss_inputs[0]->TypeAsProto();
    NodeArg& log_prob_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("log_prob"), input_def_type_proto);
    loss_outputs.emplace_back(&log_prob_def);

    Node& new_loss_node = graph.AddNode(graph.GenerateNodeName("SoftmaxCrossEntropyLossInternal"),
                                        "SoftmaxCrossEntropyLossInternal", "SoftmaxCrossEntropyLossInternal.",
                                        loss_inputs, loss_outputs, &loss_node.GetAttributes(), onnxruntime::kMSDomain);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    new_loss_node.SetExecutionProviderType(loss_node.GetExecutionProviderType());
    graph_utils::FinalizeNodeFusion(graph, new_loss_node, loss_node);
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
