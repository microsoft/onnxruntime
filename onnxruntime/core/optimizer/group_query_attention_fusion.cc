// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status GroupQueryAttentionFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {

GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (const auto& ep : GetCompatibleExecutionProviders()) {
    std::cout << std::string(ep) << std::endl;
  }

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *node_ptr;

    std::cout << node.OpType() << std::endl;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
        continue;
    }

    auto& inputs = node.MutableInputDefs();

    //  auto& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());  // get mutable next node

    std::cout << inputs.size() << std::endl;

    for (auto input : inputs) {
      std::cout << input->Name() << std::endl;
      std::cout << *input->Type() << std::endl;
    }

    std::cout << "-----------------" << std::endl;

    for (auto n = node.InputNodesBegin(); n != node.InputNodesEnd(); ++n) {
      auto& mul_node = *graph.GetNode(n->Index());  // get mutable next node

      if ((*n).OpType() == "RotaryEmbedding") {
        for (auto inner = mul_node.InputNodesBegin(); inner != mul_node.InputNodesEnd(); ++inner) {
          std::cout << "rotary input is " << (*inner).Name() << std::endl;
        }
      }
      std::cout << mul_node.Name() << std::endl;
      std::cout << mul_node.OpType() << std::endl;
    }

    for (auto* input_def : node.InputDefs()) {
      // Try to find a node that produces this input
      const Node* producer = graph.GetProducerNode(input_def->Name());
      if (producer != nullptr) {
        std::cout << "Input \"" << input_def->Name() << "\" is produced by node: " << producer->Name() << std::endl;
      } else {
        std::cout << "Input \"" << input_def->Name() << "\" is not produced by any node (it is likely an initializer or constant)" << std::endl;
      }
    }
  }
    
  return Status::OK();
}
}  // namespace onnxruntime
