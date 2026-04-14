// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/bitmask_dropout_replacement.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status BitmaskDropoutReplacement::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Matching for Dropout node.
    if ((!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {12, 13}) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(node, "BiasDropout", {1}, kMSDomain)) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (node.OutputDefs().size() < 2) continue;  // No mask output.
    const NodeArg* mask_output = node.OutputDefs()[1];

    // If mask output is used as a graph output, rewrite is impossible.
    if (graph.IsOutput(mask_output)) continue;
    auto consumer_nodes = graph.GetConsumerNodes(mask_output->Name());
    if (consumer_nodes.size() != 1 ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*consumer_nodes[0], "DropoutGrad", {1}, kMSDomain) ||
        consumer_nodes[0]->GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    Node& dropoutgrad_node = *graph.GetNode(consumer_nodes[0]->Index());

    InlinedVector<NodeArg*> dropout_output;
    dropout_output.emplace_back(node.MutableOutputDefs()[0]);
    ONNX_NAMESPACE::TypeProto tensor_uint32;
    tensor_uint32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    NodeArg& bitmask_output_def =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("dropout_bitmask_output"), &tensor_uint32);
    dropout_output.emplace_back(&bitmask_output_def);
    const std::string op_type = node.OpType() == "Dropout" ? "BitmaskDropout" : "BitmaskBiasDropout";
    Node& bitmask_dropout_node =
        graph.AddNode(graph.GenerateNodeName(op_type), op_type, "Bitmask Dropout replace for " + node.Name(),
                      node.MutableInputDefs(), dropout_output, &node.GetAttributes(), kMSDomain);
    bitmask_dropout_node.SetExecutionProviderType(node.GetExecutionProviderType());

    InlinedVector<NodeArg*> dropoutgrad_input;
    dropoutgrad_input.emplace_back(dropoutgrad_node.MutableInputDefs()[0]);
    dropoutgrad_input.emplace_back(&bitmask_output_def);
    for (size_t i = 2; i < dropoutgrad_node.InputDefs().size(); ++i) {
      dropoutgrad_input.emplace_back(dropoutgrad_node.MutableInputDefs()[i]);
    }
    const std::string grad_op_type = "BitmaskDropoutGrad";
    Node& bitmask_dropout_grad_node = graph.AddNode(
        graph.GenerateNodeName(grad_op_type), grad_op_type, "BitmaskDropoutGrad replace for " + dropoutgrad_node.Name(),
        dropoutgrad_input, dropoutgrad_node.MutableOutputDefs(), &dropoutgrad_node.GetAttributes(), kMSDomain);
    bitmask_dropout_grad_node.SetExecutionProviderType(dropoutgrad_node.GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, dropoutgrad_node);
    graph.RemoveNode(dropoutgrad_node.Index());

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
