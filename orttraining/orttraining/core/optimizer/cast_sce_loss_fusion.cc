// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/cast_sce_loss_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

Status CastSceLossFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    bool is_internal_sce = graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLossInternal", {1},
                                                                          kMSDomain);

    if (!is_internal_sce || !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    Node* input_node = graph.GetMutableProducerNode(node.MutableInputDefs()[0]->Name());

    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(*input_node, "Cast", {9, 13, 19}))) {
      continue;
    }

    if (input_node->GetOutputEdgesCount() != 1 || graph.IsOutput(input_node->OutputDefs()[0])) {
      continue;
    }

    if (input_node->MutableInputDefs()[0]->TypeAsProto()->tensor_type().elem_type() == onnx::TensorProto_DataType_FLOAT16 &&
        input_node->MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type() == onnx::TensorProto_DataType_FLOAT) {
      std::vector<graph_utils::GraphEdge> input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node, 0);
      graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edges);
      node.MutableInputDefs()[0] = input_node->MutableInputDefs()[0];
      graph_utils::MoveAllNodeInputEdges(graph, *input_node, node);
      graph.RemoveNode(input_node->Index());

      if (node.GetAttributes().count("output_type") == 0) {
        node.AddAttribute("output_type", static_cast<int64_t>(onnx::TensorProto_DataType_FLOAT));
      }
      modified = true;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
