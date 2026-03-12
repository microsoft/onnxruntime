// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_float_activations_transformer.h"

#include <vector>

#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_actions.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/selectors_actions/helpers.h"
#endif

namespace onnxruntime {

namespace {

// Remove a Q->DQ pair by bypassing both nodes and connecting Q's input source directly to DQ's downstream consumers.
// Handles graph outputs. Returns true if the pair was removed.
// This handles Q -> multiple DQ (each DQ with single consumer), following the pattern in qdq_final_cleanup.cc.
bool RemoveQDQPair(Graph& graph, Node& q_node, const logging::Logger& logger) {
  if (!QDQ::MatchQNode(q_node) || q_node.GetOutputEdgesCount() < 1) {
    return false;
  }

  // Collect all DQ consumers of Q
  std::vector<Node*> dq_nodes;
  for (auto it = q_node.OutputNodesBegin(); it != q_node.OutputNodesEnd(); ++it) {
    dq_nodes.push_back(graph.GetNode(it->Index()));
  }

  const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  // Validate: ALL consumers must be DQ nodes with matching scale/zp, each with <= 1 non-graph-output consumer
  for (auto* dq_node : dq_nodes) {
    if (!QDQ::MatchDQNode(*dq_node)) {
      return false;
    }

    if (!QDQ::IsQDQPairSupported(graph, q_node, *dq_node, get_constant_initializer,
                                 graph.ModelPath(), false)) {
      return false;
    }

    const bool produces_graph_output = graph.NodeProducesGraphOutput(*dq_node);
    const auto output_edges_count = dq_node->GetOutputEdgesCount();

    // DQ must have exactly 1 consumer, or be a graph output with no consumers
    if (produces_graph_output && output_edges_count != 0) {
      return false;
    }
    if (!produces_graph_output && output_edges_count != 1) {
      return false;
    }
  }

  LOGS(logger, VERBOSE) << "QDQFloatActivationsTransformer: removing Q node \"" << q_node.Name()
                        << "\" with " << dq_nodes.size() << " DQ consumer(s)";

  // Get Q's input edge info (source node or initializer/graph input) and remove the src->Q edge
  // upfront, before processing DQ nodes. This must happen before any modifications to
  // src_node.output_defs (e.g., in the graph-output case) which would cause a NodeArg mismatch
  // in RemoveEdge.
  NodeIndex src_node_idx = 0;
  int src_arg_idx = -1;
  if (q_node.GetInputEdgesCount() == 1) {
    const Node::EdgeEnd& input_edge = *q_node.InputEdgesBegin();
    src_node_idx = input_edge.GetNode().Index();
    src_arg_idx = input_edge.GetSrcArgIndex();
    graph.RemoveEdge(src_node_idx, q_node.Index(), src_arg_idx, 0);
  }

  // Process each DQ node
  for (auto* dq_node_ptr : dq_nodes) {
    Node& dq_node = *dq_node_ptr;
    const bool produces_graph_output = graph.NodeProducesGraphOutput(dq_node);

    // Remove edge: Q -> DQ
    graph.RemoveEdge(q_node.Index(), dq_node.Index(), 0, 0);

    if (!produces_graph_output) {
      // Get downstream consumer of DQ
      const Node::EdgeEnd& output_edge = *dq_node.OutputEdgesBegin();
      NodeIndex downstream_idx = output_edge.GetNode().Index();
      int downstream_arg_idx = output_edge.GetDstArgIndex();

      // Remove edge: DQ -> downstream
      graph.RemoveEdge(dq_node.Index(), downstream_idx, 0, downstream_arg_idx);

      // Rewire: downstream now gets Q's input
      Node& downstream_node = *graph.GetNode(downstream_idx);
      downstream_node.MutableInputDefs()[downstream_arg_idx] = q_node.MutableInputDefs()[0];

      // Add edge from Q's source to downstream (if Q's input came from a node)
      if (src_arg_idx >= 0) {
        graph.AddEdge(src_node_idx, downstream_idx, src_arg_idx, downstream_arg_idx);
      }
    } else {
      // DQ produces a graph output
      NodeArg* graph_output_nodearg = dq_node.MutableOutputDefs()[0];
      if (src_arg_idx >= 0 && dq_nodes.size() == 1) {
        // Update source node to produce the graph output
        Node& src_node = *graph.GetNode(src_node_idx);
        src_node.MutableOutputDefs()[src_arg_idx] = graph_output_nodearg;
      } else {
        // Add Identity to connect graph input/initializer to graph output
        Node& id_node = graph.AddNode(graph.GenerateNodeName("QDQFloatActivationsTransformer"),
                                      "Identity", "", {q_node.MutableInputDefs()[0]}, {graph_output_nodearg});
        id_node.SetExecutionProviderType(dq_node.GetExecutionProviderType());
      }
    }

    graph.RemoveNode(dq_node.Index());
  }

  // Q node has no edges remaining (src->Q removed upfront, Q->DQ removed in loop)
  graph.RemoveNode(q_node.Index());

  return true;
}

}  // namespace

Status QDQFloatActivationsTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                 const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Sub-pass A: Remove all adjacent Q->DQ pairs
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));

    if (QDQ::MatchQNode(*node_ptr)) {
      if (RemoveQDQPair(graph, *node_ptr, logger)) {
        modified = true;
      }
    }
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // Sub-pass B: MatMulNBits fusion for newly eligible patterns
  // After Q->DQ removal, DQ(blockwise)->MatMul patterns may now be eligible.
  // Re-get topological order since graph was modified.
  if (modified) {
    GraphViewer updated_viewer(graph);
    const auto& updated_topology = updated_viewer.GetNodesInTopologicalOrder();

    std::vector<const char*> cpu_ep = {kCpuExecutionProvider};
    QDQ::DQMatMulToMatMulNBitsSelector dq_matmul_selector(cpu_ep);
    QDQ::DQCastMatMulToMatMulNBitsSelector dq_cast_matmul_selector(cpu_ep);
    QDQ::DQMatMulToMatMulNBitsAction dq_matmul_action(qdq_matmulnbits_accuracy_level_,
                                                      intra_op_thread_pool_);
    QDQ::DQCastMatMulToMatMulNBitsAction dq_cast_matmul_action(qdq_matmulnbits_accuracy_level_,
                                                               intra_op_thread_pool_);

    for (auto node_index : updated_topology) {
      auto* node_ptr = graph.GetNode(node_index);
      if (node_ptr == nullptr || node_ptr->OpType() != "MatMul") {
        continue;
      }

      // Try DQ -> MatMul -> MatMulNBits
      auto selection = dq_matmul_selector.Select(updated_viewer, *node_ptr);
      if (selection.has_value()) {
        NodesToOptimize nto(graph, *selection);
        if (nto.IsValid()) {
          auto status = dq_matmul_action.Run(graph, nto);
          if (status.IsOK()) {
            modified = true;
            continue;
          }
          LOGS(logger, WARNING) << "QDQFloatActivationsTransformer: DQMatMulToMatMulNBits action failed: "
                                << status.ErrorMessage();
        }
      }

      // Try DQ -> Cast -> MatMul -> MatMulNBits
      auto cast_selection = dq_cast_matmul_selector.Select(updated_viewer, *node_ptr);
      if (cast_selection.has_value()) {
        NodesToOptimize nto(graph, *cast_selection);
        if (nto.IsValid()) {
          auto status = dq_cast_matmul_action.Run(graph, nto);
          if (status.IsOK()) {
            modified = true;
            continue;
          }
          LOGS(logger, WARNING) << "QDQFloatActivationsTransformer: DQCastMatMulToMatMulNBits action failed: "
                                << status.ErrorMessage();
        }
      }
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  // Sub-pass C: Constant-fold remaining weight DQ nodes.
  // After activation Q->DQ removal, weight DQ nodes on constant initializers can be folded
  // into float tensors so ops run directly on float weights.
  if (modified) {
    ConstantFolding constant_folding(cpu_execution_provider_,
                                     /*skip_dequantize_linear=*/false,
                                     config_options_);
    bool cf_modified = false;
    ORT_RETURN_IF_ERROR(constant_folding.Apply(graph, cf_modified, logger));
    modified |= cf_modified;
  }

  return Status::OK();
}

}  // namespace onnxruntime
