// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/mul_unsqueeze_concat_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MulUnsqueezeConcatFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node) {
      continue;
    }

    //std::vector<NodeArg*> fused_input_defs = {};

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Mul", {6, 7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        graph.NodeProducesGraphOutput(*node)) {
      //TODO: Make sure
      continue;
    }

    //fused_input_defs.push_back(node->MutableInputDefs()[0]);
    //fused_input_defs.push_back(node->MutableInputDefs()[1]);

    bool continue_with_fusion = true;

    // All downstream nodes to Mul must be Unsqueeze
    for (auto it1 = node->OutputNodesBegin(), end1 = node->OutputNodesEnd(); it1 != end1; ++it1) {
      if (!continue_with_fusion) {
        break;
      }

      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*it1, "Unsqueeze", {11, 13}) ||
          graph.NodeProducesGraphOutput(*it1)) {
        continue_with_fusion = false;
        break;
      }

      // All downstream nodes to Unsqueeze must be Concat
      for (auto it2 = it1->OutputNodesBegin(), end2 = it1->OutputNodesEnd(); it2 != end2; ++it2) {
        if (!graph_utils::IsSupportedOptypeVersionAndDomain(*it2, "Concat", {11, 13})) {
          continue_with_fusion = false;
          break;
        }

        // All downstream nodes to Concat must be Reshape
        for (auto it3 = it2->OutputNodesBegin(), end3 = it2->OutputNodesEnd(); it3 != end3; ++it3) {
          if (!continue_with_fusion) {
            break;
          }

          if (!graph_utils::IsSupportedOptypeVersionAndDomain(*it3, "Reshape", {14, 13, 5})) {
            continue_with_fusion = false;
            break;
          }
        }
      }
    }

    if (!continue_with_fusion) {
      continue;
    }

    // Assign Mul to CPU
    node->SetExecutionProviderType(kCpuExecutionProvider);

    // Assign Unsqueeze to CPU
    for (auto it1 = node->OutputNodesBegin(), end1 = node->OutputNodesEnd(); it1 != end1; ++it1) {
      const_cast<Node&>(*it1).SetExecutionProviderType(kCpuExecutionProvider);

      // Assign Concat to CPU
      for (auto it2 = it1->OutputNodesBegin(), end2 = it1->OutputNodesEnd(); it2 != end2; ++it2) {
        const_cast<Node&>(*it2).SetExecutionProviderType(kCpuExecutionProvider);
      }
    }

    // Assign all Mul, Unsqueeze, and Concat to CPU

    /*
    auto& first_concat_node = const_cast<Node&>(*node->OutputNodesBegin()->OutputNodesBegin());
    fused_input_defs.insert(fused_input_defs.begin(), first_concat_node.MutableInputDefs()[0]);
    fused_input_defs.push_back(first_concat_node.MutableInputDefs()[2]);

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("MulUnsqueezeConcatFused_" + first_concat_node.Name()),
                                     "MulUnsqueezeConcatFused",
                                     "fused mul, unsqueeze, concat ",
                                     fused_input_defs,
                                     {first_concat_node.MutableOutputDefs()[0]},
                                     {},
                                     kMSDomain);

    fused_node.SetExecutionProviderType(kCpuExecutionProvider);


    *     // Adjust edges and remove nodes

    std::vector<const Node*> nodes_to_be_removed;

    std::vector<NodeIndex> dst_nodes;
    std::vector<int> dst_node_args;

    for (auto it1 = node->OutputNodesBegin(), end1 = node->OutputNodesEnd(); it1 != end1; ++it1) {
      for (auto it2 = it1->OutputNodesBegin(), end2 = it1->OutputNodesEnd(); it2 != end2; ++it2) {
        const auto& concat_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*it2);

        // Note down destination edges from Concat
        for (auto& edge : concat_output_edges) {
          dst_nodes.push_back(edge.dst_node);
          dst_node_args.push_back(edge.dst_arg_index);
        }

        //  TODO: Adjust input edges if needed
        graph_utils::GraphEdge::RemoveGraphEdges(graph, graph_utils::GraphEdge::GetNodeOutputEdges(*it2));
        nodes_to_be_removed.push_back(&*it2);
      }

      graph_utils::GraphEdge::RemoveGraphEdges(graph, graph_utils::GraphEdge::GetNodeOutputEdges(*it1));
      nodes_to_be_removed.push_back(&*it1);
    }

    // Add edges from fused node
    for (int64_t i = 0; i < static_cast<int64_t>(dst_nodes.size()); ++i) {
      graph.AddEdge(fused_node.Index(),
                    dst_nodes[i],
                    0,
                    dst_node_args[i]);
    }

    // Remove root Mul node
    graph_utils::GraphEdge::RemoveGraphEdges(graph, graph_utils::GraphEdge::GetNodeOutputEdges(*node));
    graph.RemoveNode(node->Index());

    // Remove nodes
    for (auto* node_to_be_removed : nodes_to_be_removed) {
      graph.RemoveNode(node_to_be_removed->Index());
    }
  }
  */
  }

  return Status::OK();
}

}  // namespace onnxruntime
