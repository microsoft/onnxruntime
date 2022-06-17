// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/cast_gelu_fusion.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace {
bool IsCastBetweenHalfAndFloat(const Graph& graph, const Node& node, bool is_reverse) {
  if (node.OpType() != "Cast") return false;
  const auto* output_def = node.OutputDefs()[0];
  if (graph.IsOutput(output_def)) return false;
  const auto* input_type_proto = node.InputDefs()[0]->TypeAsProto();
  const auto* output_type_proto = output_def->TypeAsProto();
  if (!input_type_proto || !output_type_proto) return false;
  int input_type = input_type_proto->tensor_type().elem_type();
  int output_type = output_type_proto->tensor_type().elem_type();
  return ((!is_reverse && input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
           output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
          (is_reverse && input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
           output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
}
}  // namespace

/**
CastGeluFusion will Remove all the Cast node for 2 cases below:

X -> Cast(half to float) -> Gelu -> Cast(float to half) -> Y

TO

X -> Gelu -> Y

OR

                        X -> Cast(half to float) -> Gelu -> Cast(float to half) -> Y
                              |
dY -> Cast(half to float) -> GeluGrad -> Cast(float to half) -> dX

TO

       X  -> Gelu -> Y
       |
dY -> GeluGrad -> dX
*/
Status CastGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // The first Cast must be from half to float.
    if (!IsCastBetweenHalfAndFloat(graph, node, false) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
    nodes_to_remove.emplace_back(node);

    bool can_fuse = true;
    Node* p_gelu_node = nullptr;
    Node* p_gelu_grad_node = nullptr;
    for (const Node* p_node : graph.GetConsumerNodes(node.OutputDefs()[0]->Name())) {
      Node& next_node = *graph.GetNode(p_node->Index());
      if (next_node.OpType() == "Gelu") {
        // There must be one Gelu node consumes the Cast output. This node doesn't produce graph output,
        // and has one and only one Cast node from float to half as the output consumer.
        if (p_gelu_node) {
          can_fuse = false;
          break;
        }
        p_gelu_node = &next_node;
        auto* output_def = next_node.OutputDefs()[0];
        auto gelu_consumer_nodes = graph.GetConsumerNodes(output_def->Name());
        if (graph.IsOutput(output_def) || gelu_consumer_nodes.size() != 1) {
          can_fuse = false;
          break;
        }
        Node& consumer_node = *graph.GetNode(gelu_consumer_nodes[0]->Index());
        if (!IsCastBetweenHalfAndFloat(graph, consumer_node, true) ||
            !graph_utils::IsSupportedProvider(consumer_node, GetCompatibleExecutionProviders())) {
          can_fuse = false;
          break;
        }
        nodes_to_remove.emplace_back(consumer_node);
      } else if (next_node.OpType() == "GeluGrad") {
        if (p_gelu_grad_node) {
          can_fuse = false;
          break;
        }
        p_gelu_grad_node = &next_node;
        Node& producer_node = *graph.GetNode(graph.GetProducerNode(next_node.InputDefs()[0]->Name())->Index());
        Node& consumer_node = *graph.GetNode(graph.GetConsumerNodes(next_node.OutputDefs()[0]->Name())[0]->Index());
        if (!IsCastBetweenHalfAndFloat(graph, producer_node, false) ||
            !IsCastBetweenHalfAndFloat(graph, consumer_node, true)) {
          can_fuse = false;
          break;
        }
        nodes_to_remove.emplace_back(producer_node);
        nodes_to_remove.emplace_back(consumer_node);
      } else {
        can_fuse = false;
        break;
      }
    }

    if (!can_fuse || !p_gelu_node) continue;

    const auto& type_proto = *node.InputDefs()[0]->TypeAsProto();
    graph.SetNodeArgType(*p_gelu_node->MutableInputDefs()[0], type_proto);
    graph.SetNodeArgType(*p_gelu_node->MutableOutputDefs()[0], type_proto);
    if (p_gelu_grad_node) {
      graph.SetNodeArgType(*p_gelu_grad_node->MutableInputDefs()[0], type_proto);
      graph.SetNodeArgType(*p_gelu_grad_node->MutableInputDefs()[1], type_proto);
      graph.SetNodeArgType(*p_gelu_grad_node->MutableOutputDefs()[0], type_proto);
    }

    for (Node& n : nodes_to_remove) {
      graph_utils::RemoveNode(graph, n);
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
