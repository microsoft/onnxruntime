// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace {
// type of node sequence to clean up
enum class NodeSequence {
  Q_DQ,
  DQ_Q,
};

bool CleanUpNodeSequence(NodeSequence node_sequence_type, Graph& graph, NodeIndex first_node_idx,
                         const logging::Logger& logger) {
  Node* first_node_ptr = graph.GetNode(first_node_idx);
  if (!first_node_ptr) {
    return false;
  }
  Node& first_node = *first_node_ptr;

  const auto match_first = node_sequence_type == NodeSequence::Q_DQ ? &QDQ::MatchQNode : &QDQ::MatchDQNode;
  const auto match_second = node_sequence_type == NodeSequence::Q_DQ ? &QDQ::MatchDQNode : &QDQ::MatchQNode;

  if (!match_first(first_node) ||
      // not filtering on provider currently
      // !graph_utils::IsSupportedProvider(first_node, compatible_execution_providers) ||
      !optimizer_utils::CheckOutputEdges(graph, first_node, 1)) {
    return false;
  }

  Node& second_node = *graph.GetNode(first_node.OutputNodesBegin()->Index());
  if (!match_second(second_node)
      // not filtering on provider currently
      // || !graph_utils::IsSupportedProvider(second_node, compatible_execution_providers)
  ) {
    return false;
  }

  if (node_sequence_type == NodeSequence::DQ_Q) {
    // for DQ -> Q, check for constant, matching scale/ZP values
    const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
      return graph.GetConstantInitializer(initializer_name, true);
    };

    if (!QDQ::IsQDQPairSupported(second_node, first_node, get_constant_initializer, graph.ModelPath())) {
      return false;
    }
  }

  // src node or graph input/initializer -> first_node -> second_node -> downstream node(s) or graph output

  // we support these cases
  // - second_node produces a graph output and has no output edges
  // - second_node does not produce a graph output
  const bool produces_graph_output = graph.NodeProducesGraphOutput(second_node);
  const bool has_output_edges = second_node.GetOutputEdgesCount() > 0;
  if (produces_graph_output && has_output_edges) {
    return false;
  }

  // we have a node sequence to clean up
  LOGS(logger, VERBOSE) << "Cleaning up back-to-back nodes: "
                        << first_node.OpType() << " with name \"" << first_node.Name() << "\" and "
                        << second_node.OpType() << " with name \"" << second_node.Name() << "\"";

  const bool has_src_node = first_node.GetInputEdgesCount() == 1;
  NodeIndex src_node_idx = 0;
  int src_arg_idx = -1;

  // input could be node or initializer/graph input so need to handle both.
  // if it's a node we need to replace the edge, so need info on which output idx it was attached to on the src node.
  if (has_src_node) {
    const Node::EdgeEnd* input_edge = &*first_node.InputEdgesBegin();
    src_node_idx = input_edge->GetNode().Index();
    src_arg_idx = input_edge->GetSrcArgIndex();
    // remove edge from src to first_node. dest arg idx is 0 as first_node (Q or DQ) only has one input
    graph.RemoveEdge(src_node_idx, first_node.Index(), src_arg_idx, 0);
  }

  // remove edge between pair we're removing
  // both DQ and Q are single input single output so src idx and dest idx must be 0
  graph.RemoveEdge(first_node.Index(), second_node.Index(), 0, 0);

  if (!produces_graph_output) {
    if (has_output_edges) {
      // make a copy of second_node's output EdgeEnds since we will remove them from second_node as we go
      const auto copied_output_edges = InlinedVector<Node::EdgeEnd>(second_node.OutputEdgesBegin(),
                                                                    second_node.OutputEdgesEnd());
      for (const auto& output_edge : copied_output_edges) {
        // remove edge to downstream node
        const auto downstream_node_idx = output_edge.GetNode().Index();
        const auto downstream_arg_idx = output_edge.GetDstArgIndex();

        // source arg idx is 0 as Q/DQ only has one output
        graph.RemoveEdge(second_node.Index(), downstream_node_idx, 0, downstream_arg_idx);

        // replace input on downstream node
        Node& downstream_node = *graph.GetNode(downstream_node_idx);
        downstream_node.MutableInputDefs()[downstream_arg_idx] = first_node.MutableInputDefs()[0];

        // create edge between src_node (if available) and downstream node
        if (has_src_node) {
          graph.AddEdge(src_node_idx, downstream_node_idx, src_arg_idx, downstream_arg_idx);
        }
      }
    }
  } else {
    NodeArg* graph_output_nodearg = second_node.MutableOutputDefs()[0];
    if (has_src_node) {
      // update the src node to produce the graph output that was being provided by second_node
      Node& src_node = *graph.GetNode(src_node_idx);
      src_node.MutableOutputDefs()[src_arg_idx] = graph_output_nodearg;
    } else {
      // add Identity node to connect the graph input or initializer to the graph output.
      Node& id_node = graph.AddNode(graph.GenerateNodeName("QDQFinalCleanupTransformer"),
                                    "Identity", "", {first_node.MutableInputDefs()[0]}, {graph_output_nodearg});
      id_node.SetExecutionProviderType(second_node.GetExecutionProviderType());
    }
  }

  graph.RemoveNode(first_node.Index());
  graph.RemoveNode(second_node.Index());

  return true;
}
}  // namespace

Status QDQFinalCleanupTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node was removed as part of an earlier optimization

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (CleanUpNodeSequence(NodeSequence::DQ_Q, graph, node_index, logger)) {
      modified = true;
    }

    if (enable_q_dq_cleanup_) {
      if (CleanUpNodeSequence(NodeSequence::Q_DQ, graph, node_index, logger)) {
        modified = true;
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
