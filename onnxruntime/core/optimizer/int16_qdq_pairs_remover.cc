// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/int16_qdq_pairs_remover.h"
#include <cassert>
#include <string>

#include "core/common/span_utils.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

bool Int16QDQPairsRemover::TryRemoveInt16QDQPairs(Graph& graph, NodeIndex quantize_node_index) const {
  const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  Node* quantize_node = graph.GetNode(quantize_node_index);

  if (quantize_node == nullptr ||
      !graph_utils::IsSupportedProvider(*quantize_node, GetCompatibleExecutionProviders()) ||
      quantize_node->OpType() != "QuantizeLinear" ||
      graph.NodeProducesGraphOutput(*quantize_node)) {
    return false;
  }

  auto quant_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (!QDQ::GetQNodeZeroPointType(graph, *quantize_node, quant_type)) {
    return false;
  }

  // Ensure that the quantization type is int16 or uint16
  if (quant_type != ONNX_NAMESPACE::TensorProto_DataType_UINT16 && quant_type != ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    return false;
  }

  // All of q2's children should be DQ nodes with zero-point and scale values equal to those of q2.
  InlinedVector<gsl::not_null<Node*>> dequantize_nodes;
  dequantize_nodes.reserve(quantize_node->GetOutputEdgesCount());

  for (auto it = quantize_node->OutputEdgesBegin(); it != quantize_node->OutputEdgesEnd(); it++) {
    NodeIndex dequantize_node_index = it->GetNode().Index();
    Node* dequantize_node = graph.GetNode(dequantize_node_index);

    if (dequantize_node == nullptr || dequantize_node->OpType() != "DequantizeLinear") {
      // Child is not a DQ op.
      return false;
    }

    // The Q2 and DQ2 nodes must have equal zero-point and scale values (scalar/constant).
    if (!QDQ::IsQDQPairSupported(*quantize_node, *dequantize_node, get_constant_initializer, graph.ModelPath())) {
      return false;
    }

    dequantize_nodes.push_back(dequantize_node);
  }

  const Node* source_node = graph.GetProducerNode(quantize_node->InputDefs()[0]->Name());
  const int source_node_output_index = source_node ? quantize_node->InputEdgesBegin()->GetSrcArgIndex() : -1;

  auto input_node_arg = quantize_node->MutableInputDefs()[0];
  auto constant_initializer = graph.GetConstantInitializer(input_node_arg->Name(), true);

  NodeArg* constant_initializer_node_arg = nullptr;

  if (constant_initializer) {
    auto initializer_copy = *constant_initializer;
    initializer_copy.set_name(constant_initializer->name() + "_copy");
    constant_initializer_node_arg = &graph_utils::AddInitializer(graph, initializer_copy);
  }

  // Disconnect the source node from the quantize node
  if (source_node) {
    graph.RemoveEdge(source_node->Index(), quantize_node_index, source_node_output_index, 0);
  }

  // Disconnect the quantize node from the dequantize nodes, and connect the source node to the outputs of dequantize
  for (gsl::not_null<Node*> dequantize_node : dequantize_nodes) {
    graph.RemoveEdge(quantize_node_index, dequantize_node->Index(), 0, 0);

    if (source_node && dequantize_node->GetOutputEdgesCount() == 0) {
      graph.GetNode(source_node->Index())->MutableOutputDefs()[source_node_output_index] = dequantize_node->MutableOutputDefs()[0];
    }

    while (dequantize_node->GetOutputEdgesCount() > 0) {
      auto output_iter = dequantize_node->OutputEdgesBegin();
      auto target_node = graph.GetNode(output_iter->GetNode().Index());
      const int target_arg_index = output_iter->GetDstArgIndex();

      graph.RemoveEdge(dequantize_node->Index(), target_node->Index(), output_iter->GetSrcArgIndex(), target_arg_index);

      if (source_node) {
        graph.AddEdge(source_node->Index(), target_node->Index(), source_node_output_index, target_arg_index);
      } else {
        if (constant_initializer_node_arg) {
          target_node->MutableInputDefs()[target_arg_index] = constant_initializer_node_arg;
        } else {
          target_node->MutableInputDefs()[target_arg_index] = input_node_arg;
        }
      }
    }

    graph.RemoveNode(dequantize_node->Index());
  }

  graph.RemoveNode(quantize_node_index);

  return true;
}

Status Int16QDQPairsRemover::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  const GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex node_index : node_topology_list) {
    if (TryRemoveInt16QDQPairs(graph, node_index)) {
      modified = true;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
