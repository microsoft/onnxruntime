// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status BiasGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
      continue;
    }

    InlinedVector<NodeArg*> gelu_input;
    const TensorShapeProto* input1_shape = node.MutableInputDefs()[0]->Shape();
    const TensorShapeProto* input2_shape = node.MutableInputDefs()[1]->Shape();

    if (input1_shape == nullptr ||
        input2_shape == nullptr ||
        input1_shape->dim_size() < 1 ||
        input2_shape->dim_size() < 1) {
      continue;
    }

    if (input1_shape->dim(input1_shape->dim_size() - 1) != input2_shape->dim(input2_shape->dim_size() - 1)) {
      continue;
    }

    if (input1_shape->dim_size() == 1) {
      gelu_input.push_back(node.MutableInputDefs()[1]);
      gelu_input.push_back(node.MutableInputDefs()[0]);
    } else if (input2_shape->dim_size() == 1) {
      gelu_input.push_back(node.MutableInputDefs()[0]);
      gelu_input.push_back(node.MutableInputDefs()[1]);
    } else {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Gelu", {1}, kMSDomain) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "FastGelu", {1}, kMSDomain)) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    bool is_fast_gelu = next_node.OpType().compare("FastGelu") == 0;
    if (is_fast_gelu && next_node.InputDefs().size() > 1) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    Node& add_node = node;
    Node& gelu_node = const_cast<Node&>(next_node);
    std::string op_type = "BiasGelu";
    if (is_fast_gelu) op_type = "FastGelu";

    Node& gelu_add_fusion_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                               op_type,
                                               "fused Add and Gelu",
                                               gelu_input,
                                               {},
                                               {},
                                               kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_add_fusion_node.SetExecutionProviderType(gelu_node.GetExecutionProviderType());

    // move output definitions and edges from gelu_node to gelu_add_fusion_node
    // delete add_node and gelu_node.
    graph_utils::FinalizeNodeFusion(graph, {add_node, gelu_node}, gelu_add_fusion_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
