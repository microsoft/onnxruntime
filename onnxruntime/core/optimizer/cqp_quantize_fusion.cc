// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cqp_quantize_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

/**
CQPQuantizeLinearFusion will fuse subgraph like below into DynamicQuantizeLinear:
 
ComputeQuantizationParameters                     
            |
            |                          ------>    DynamicQuantizeLinear    
            v
      QuantizeLinear

 */
Status CQPQuantizeLinearFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& quant_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(quant_node, modified, graph_level, logger));

    // V ARE THE OPTYPES CORRECT V
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(quant_node, "QuantizeLinear", {10, 13}) ||
        !graph_utils::IsSupportedProvider(quant_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    const Node* p_cqp_node = graph_utils::FirstParentByType(quant_node, "ComputeQuantizationParameters");
    if (p_cqp_node == nullptr) {
      continue;
    }

    Node& cqp_node = *graph.GetNode(p_cqp_node->Index());

    // Check Nodes' Edges count and Nodes' outputs are not in Graph output
    if (!optimizer_utils::CheckOutputEdges(graph, cqp_node, 1)) {
      continue;
    }

    // ComputeQuantizationParameters outputs are only used by one QuantizeLinear,
    // thus it can fused into DynamicQuantizeLinear. <<< IS THIS TRUE?
    NodeArg optional_node_arg("", nullptr);
    InlinedVector<NodeArg*> input_defs{
        cqp_node.MutableInputDefs()[0]
    };

    std::string op_type = "DynamicQuantizeLinear";
    Node& fused_node = graph.AddNode(quant_node.Name(),
                                     op_type,
                                     "",
                                     input_defs,
                                     quant_node.MutableOutputDefs(),
                                     nullptr,
                                     kMSDomain);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(quant_node.GetExecutionProviderType());

    nodes_to_remove.push_back(quant_node);
    nodes_to_remove.push_back(cqp_node);
  }

  modified = modified || !nodes_to_remove.empty();

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  return Status::OK();
}
}  // namespace onnxruntime
