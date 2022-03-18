// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
Status QDQFinalCleanupTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node was removed as part of an earlier optimization

    Node& q_node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(q_node, modified, graph_level, logger));

    if (!QDQ::MatchQNode(q_node) ||
        // not filtering on provider currently
        // !graph_utils::IsSupportedProvider(q_node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, q_node, 1)) {
      continue;
    }

    Node& dq_node = *graph.GetNode(q_node.OutputNodesBegin()->Index());
    if (!QDQ::MatchDQNode(dq_node) 
        // not filtering on provider currently
        // || !graph_utils::IsSupportedProvider(dq_node, GetCompatibleExecutionProviders())
        ) {
      continue;
    }

    // we have Q -> DQ pair

    // we support a DQ that produces a graph output if it has no output edges, or a DQ node with one output edge.
    bool is_graph_output = graph.NodeProducesGraphOutput(dq_node);
    auto edges_count = dq_node.GetOutputEdgesCount();

    if ((is_graph_output && edges_count != 0) ||
        (!is_graph_output && edges_count != 1)) {
      continue;
    }

    // src node -> Q -> DQ -> downstream node or graph output
    NodeIndex src_node_idx = 0;
    int src_arg_idx = -1;
    NodeIndex downstream_node_idx = 0;
    int downstream_arg_idx = -1;

    // input could be node or initializer/graph input so need to handle both.
    // if it's a node we need to replace the edge, so need info on which output idx it was attached to on the src node.
    const Node::EdgeEnd* input_edge = nullptr;
    if (q_node.GetInputEdgesCount() == 1) {
      input_edge = &*q_node.InputEdgesBegin();
      src_node_idx = input_edge->GetNode().Index();
      src_arg_idx = input_edge->GetSrcArgIndex();
      // remove edge from src to Q. dest arg idx is 0 as Q only has one input
      graph.RemoveEdge(src_node_idx, q_node.Index(), src_arg_idx, 0);
    }

    // remove edge between pair we're removing
    // both DQ and Q are single input single output so src idx and dest idx must be 0
    graph.RemoveEdge(q_node.Index(), dq_node.Index(), 0, 0);

    if (!is_graph_output) {
      // remove edge to downstream node
      const Node::EdgeEnd& output_edge = *dq_node.OutputEdgesBegin();
      downstream_node_idx = output_edge.GetNode().Index();
      downstream_arg_idx = output_edge.GetDstArgIndex();

      // source arg idx is 0 as Q/DQ only has one output
      graph.RemoveEdge(dq_node.Index(), downstream_node_idx, 0, downstream_arg_idx);

      // replace input on downstream node
      Node& downstream_node = *graph.GetNode(downstream_node_idx);
      downstream_node.MutableInputDefs()[downstream_arg_idx] = q_node.MutableInputDefs()[0];

      // create edge between src_node (if available) and downstream node
      if (input_edge) {
        graph.AddEdge(src_node_idx, downstream_node_idx, src_arg_idx, downstream_arg_idx);
      }
    } else {
      NodeArg* graph_output_nodearg = dq_node.MutableOutputDefs()[0];
      if (src_arg_idx >= 0) {
        // update the src node to produce the graph output that was being provided by the DQ node
        Node& src_node = *graph.GetNode(src_node_idx);
        src_node.MutableOutputDefs()[src_arg_idx] = graph_output_nodearg;
      } else {
        // add Identity node to connect the graph input or initializer to the graph output.
        Node& id_node = graph.AddNode(graph.GenerateNodeName("QDQFinalCleanupTransformer"),
                                      "Identity", "", {q_node.MutableInputDefs()[0]}, {graph_output_nodearg});
        id_node.SetExecutionProviderType(dq_node.GetExecutionProviderType());
      }
    }

    graph.RemoveNode(q_node.Index());
    graph.RemoveNode(dq_node.Index());

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
