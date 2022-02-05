// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_propagation.h"

#include <optional>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace {

// like graph_utils::GraphEdge, but also allowing the source end to be an input and the destination end to be an output
struct PropagationEdge {
  enum class End { Source, Destination };

  struct NodeInfo {
    NodeIndex node_idx;
    int arg_idx;
  };

  std::optional<NodeInfo> src;  // if empty, graph input
  std::optional<NodeInfo> dst;  // if empty, graph output
  std::string arg_name;

  bool HasGraphInputOrInitializer() const noexcept { return !src.has_value(); }
  bool HasGraphOutput() const noexcept { return !dst.has_value(); }

  const std::optional<NodeInfo>& GetNodeInfoAtEnd(End end) const {
    return end == End::Source ? src : dst;
  }

  Node* GetNodeAtEnd(Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  const Node* GetNodeAtEnd(const Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      const Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  static PropagationEdge FromGraphEdge(const graph_utils::GraphEdge& graph_edge) {
    return PropagationEdge{
        NodeInfo{graph_edge.src_node, graph_edge.src_arg_index},
        NodeInfo{graph_edge.dst_node, graph_edge.dst_arg_index},
        graph_edge.arg_name};
  }

  static std::optional<PropagationEdge> FromInputOrInitializerToNode(
      const Graph& graph, const Node& node, int node_input_def_idx) {
    const auto node_inputs = node.InputDefs();
    ORT_ENFORCE(node_input_def_idx <= node_inputs.size());

    const auto* node_input = node_inputs[node_input_def_idx];
    if (!graph.IsInputsIncludingInitializers(node_input)) {
      return std::nullopt;
    }

    return PropagationEdge{
        std::nullopt,
        NodeInfo{node.Index(), node_input_def_idx},
        node_input->Name()};
  }

  static std::optional<PropagationEdge> FromNodeToOutput(
      const Graph& graph, const Node& node, int node_output_def_idx) {
    const auto node_outputs = node.OutputDefs();
    ORT_ENFORCE(node_output_def_idx <= node_outputs.size());

    const auto* node_output = node_outputs[node_output_def_idx];
    if (!graph.IsOutput(node_output)) {
      return std::nullopt;
    }

    return PropagationEdge{
        NodeInfo{node.Index(), node_output_def_idx},
        std::nullopt,
        node_output->Name()};
  }

  // there is also the case where the graph input is an output, but we don't care about that for this transformer
};

bool CanNodePropagate(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {5, 13, 14}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13});
}

static bool IsSupportedQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "QuantizeLinear", {10, 13});
}

static bool IsSupportedDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "DequantizeLinear", {10, 13});
}

//static bool TryCancelOutDQQPair(Graph& graph, Node& dq_node, Node& q_node) {
//  auto get_const_initializer = [&graph](const std::string& initializer_name) {
//    return graph.GetConstantInitializer(initializer_name, true);
//  };
//
//  if (!QDQ::IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph.ModelPath())) {
//    return false;
//  }
//
//  // check if dq_node has only one output edge and,
//  // dq_node and q_node output are not graph outputs
//  if (!optimizer_utils::CheckOutputEdges(graph, dq_node, 1) ||
//      graph.NodeProducesGraphOutput(q_node)) {
//    return false;
//  }
//
//  // remove edge between parent of DQ to DQ
//  std::pair<NodeIndex, int> input_edge_info{0, -1};
//  auto* dq_input_edge_0 = graph_utils::GetInputEdge(dq_node, 0);
//  if (dq_input_edge_0) {
//    input_edge_info.first = dq_input_edge_0->GetNode().Index();
//    input_edge_info.second = dq_input_edge_0->GetSrcArgIndex();
//    graph.RemoveEdge(dq_input_edge_0->GetNode().Index(), dq_node.Index(),
//                     dq_input_edge_0->GetSrcArgIndex(), dq_input_edge_0->GetDstArgIndex());
//  }
//
//  graph_utils::RemoveNodeOutputEdges(graph, dq_node);  // Remove DQ node output edges
//
//  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(q_node, 0);
//  graph_utils::RemoveNodeOutputEdges(graph, q_node);  // Remove Q node output edges
//  for (auto& output_edge : output_edges) {
//    // set input NodeArg of Q's children to the 1st input of DQ
//    graph.GetNode(output_edge.dst_node)->MutableInputDefs()[output_edge.dst_arg_index] =
//        dq_node.MutableInputDefs()[0];
//
//    // add edge between parent of DQ to children of Q
//    if (input_edge_info.second != -1) {
//      graph.AddEdge(input_edge_info.first, output_edge.dst_node,
//                    input_edge_info.second, output_edge.dst_arg_index);
//    }
//  }
//
//  graph.RemoveNode(dq_node.Index());
//  graph.RemoveNode(q_node.Index());
//  return true;
//}

// A helper function that swap the relationship of 2 nodes.
// @param up_node The original parent node
// @param down_node The original child node
// after calling the function, the up_node will become the child of down_node
// Assumptions of this function are:
// 1. up_node only has one Edge that points to down_node and its output is not graph output.
// 2. NodeArg slots of the edge between up_node and down_node are (0, 0)
//static void SwapAdjacentNodes(Graph& graph, Node& up_node, Node& down_node) {
//  ORT_ENFORCE(optimizer_utils::CheckOutputEdges(graph, up_node, 1),
//              "up_node should have only one Edge that points to down_node and its output is not graph output");
//
//  auto edge_it = up_node.OutputEdgesBegin();
//  ORT_ENFORCE(edge_it->GetDstArgIndex() == 0 &&
//                  edge_it->GetSrcArgIndex() == 0 &&
//                  edge_it->GetNode().Index() == down_node.Index(),
//              "up_node should be parent of down_node and NodeArg slots of the edge between up_node and down_node should be (0, 0).");
//
//  // ************** Remove edges **************
//  // Remove the edge between parent of up_node and up_node, and keep the info of parent of up_node
//  std::pair<NodeIndex, int> up_node_input_edge_info{0, -1};
//  auto* up_node_input_edge_0 = graph_utils::GetInputEdge(up_node, 0);
//  if (up_node_input_edge_0) {
//    up_node_input_edge_info.first = up_node_input_edge_0->GetNode().Index();
//    up_node_input_edge_info.second = up_node_input_edge_0->GetSrcArgIndex();
//    graph.RemoveEdge(up_node_input_edge_0->GetNode().Index(),
//                     up_node.Index(),
//                     up_node_input_edge_0->GetSrcArgIndex(),
//                     up_node_input_edge_0->GetDstArgIndex());
//  }
//
//  auto down_node_output_edges_info = graph_utils::GraphEdge::GetNodeOutputEdges(down_node);
//  graph_utils::RemoveNodeOutputEdges(graph, up_node);
//  graph_utils::RemoveNodeOutputEdges(graph, down_node);
//
//  // *********** Rebuild NodeArg ****************/
//  down_node.MutableInputDefs()[0] = up_node.MutableInputDefs()[0];
//  up_node.MutableOutputDefs()[0] = down_node.MutableOutputDefs()[0];
//
//  NodeArg* new_node_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SwapAdjacentNodes"), nullptr);
//  down_node.MutableOutputDefs()[0] = new_node_arg;
//  up_node.MutableInputDefs()[0] = new_node_arg;
//
//  // *********** Rebuild Edges ****************/
//  if (up_node_input_edge_info.second >= 0) {
//    graph.AddEdge(up_node_input_edge_info.first, down_node.Index(), up_node_input_edge_info.second, 0);
//  }
//
//  graph.AddEdge(down_node.Index(), up_node.Index(), 0, 0);
//
//  for (auto output_edge_info : down_node_output_edges_info) {
//    graph.AddEdge(up_node.Index(), output_edge_info.dst_node, 0, output_edge_info.dst_arg_index);
//  }
//}

// convert this: src_node -> dst_node
// to this:      src_node -> Q -> DQ -> dst_node
// assumptions:
// 1. insertion_edge is valid - node indexes refer to valid nodes and the arg name refers to a valid NodeArg
// 2. scale_initializer_nodearg and zp_initializer_nodearg are constant initializers
static Status InsertQDQPair(Graph& graph, const PropagationEdge& insertion_edge,
                            NodeArg& scale_initializer_nodearg,
                            NodeArg& zp_initializer_nodearg, const logging::Logger& logger) {
  auto* src_node = insertion_edge.GetNodeAtEnd(graph, PropagationEdge::End::Source);
  auto* dst_node = insertion_edge.GetNodeAtEnd(graph, PropagationEdge::End::Destination);

  ORT_ENFORCE(src_node || dst_node, "At least one graph node must be specified in the propagation edge.");

  const auto& base_name = insertion_edge.arg_name;
  auto& base_node_arg = *graph.GetNodeArg(base_name);

  LOGS(logger, VERBOSE) << "Inserting Q/DQ pair between "
                        << (src_node ? MakeString("node \"", src_node->Name(), "\" ") : "input ")
                        << "and "
                        << (dst_node ? MakeString("node \"", dst_node->Name(), "\" ") : "output ")
                        << " at NodeArg \"" << base_name << "\".";

  // set up new NodeArgs
  auto& pre_q_nodearg = insertion_edge.HasGraphInputOrInitializer()
                            ? base_node_arg
                            : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_pre_q"),
                                                       nullptr);

  auto& q_to_dq_nodearg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_q_to_dq"),
                                                   nullptr);

  auto& post_dq_nodearg = insertion_edge.HasGraphOutput()
                              ? base_node_arg
                              : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_post_dq"),
                                                         nullptr);

  // set up new Nodes
  auto& q_node = graph.AddNode(graph.GenerateNodeName(base_name + "_q"),
                               "QuantizeLinear",
                               "Inserted by QDQPropagationTransformer",
                               // inputs
                               {&pre_q_nodearg,
                                &scale_initializer_nodearg,
                                &zp_initializer_nodearg},
                               // outputs
                               {&q_to_dq_nodearg});

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(q_node), "Failed to set op schema for added Q node.");

  auto& dq_node = graph.AddNode(graph.GenerateNodeName(base_name + "_dq"),
                                "DequantizeLinear",
                                "Inserted by QDQPropagationTransformer",
                                // inputs
                                {&q_to_dq_nodearg, &scale_initializer_nodearg, &zp_initializer_nodearg},
                                // outputs
                                {&post_dq_nodearg});

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(dq_node), "Failed to set op schema for added DQ node.");

  // set up edges
  if (src_node && dst_node) {
    graph.RemoveEdge(src_node->Index(), dst_node->Index(),
                     insertion_edge.src->arg_idx, insertion_edge.dst->arg_idx);
  }

  if (src_node) {
    src_node->MutableOutputDefs()[insertion_edge.src->arg_idx] = &pre_q_nodearg;
    graph.AddEdge(src_node->Index(), q_node.Index(), insertion_edge.src->arg_idx, 0);
  }

  graph.AddEdge(q_node.Index(), dq_node.Index(), 0, 0);

  if (dst_node) {
    dst_node->MutableInputDefs()[insertion_edge.dst->arg_idx] = &post_dq_nodearg;
    graph.AddEdge(dq_node.Index(), dst_node->Index(), 0, insertion_edge.dst->arg_idx);
  }

  return Status::OK();
}

static bool IsMatchingQDQPair(const Graph& graph, const Node& q_node, const Node& dq_node) {
  return IsSupportedDQNode(dq_node) && IsSupportedQNode(q_node) &&
         QDQ::IsQDQPairSupported(
             q_node, dq_node,
             [&graph](const std::string& name) { return graph.GetConstantInitializer(name, true); },
             graph.ModelPath());
}

static std::optional<PropagationEdge> GetPreviousPropagationEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first input (index 0)

  const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
  const auto input_edge_it = std::find_if(
      input_edges.begin(), input_edges.end(),
      [](const graph_utils::GraphEdge& edge) { return edge.dst_arg_index == 0; });

  if (input_edge_it == input_edges.end()) {
    // maybe edge from input
    return PropagationEdge::FromInputOrInitializerToNode(graph, node, 0);
  }

  const auto& src_node = *graph.GetNode(input_edge_it->src_node);
  const auto src_node_output_edges =
      graph_utils::GraphEdge::GetNodeOutputEdges(src_node, input_edge_it->src_arg_index);
  if (!graph.IsOutput(src_node.OutputDefs()[input_edge_it->src_arg_index]) &&
      src_node_output_edges.size() == 1) {
    // single edge from previous node
    return PropagationEdge::FromGraphEdge(*input_edge_it);
  }

  return std::nullopt;
}

static std::optional<PropagationEdge> GetPreviousPropagationEdge(const Graph& graph,
                                                                 const PropagationEdge& edge) {
  if (edge.HasGraphInputOrInitializer()) {
    return std::nullopt;
  }

  return GetPreviousPropagationEdge(graph, *edge.GetNodeAtEnd(graph, PropagationEdge::End::Source));
}

static std::optional<PropagationEdge> GetNextPropagationEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first output (index 0)

  const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node, 0);
  if (output_edges.empty()) {
    // maybe edge to output
    return PropagationEdge::FromNodeToOutput(graph, node, 0);
  }

  if (!graph.IsOutput(node.OutputDefs()[0]) && output_edges.size() == 1) {
    // single edge to next node
    return PropagationEdge::FromGraphEdge(output_edges.front());
  }

  return std::nullopt;
}

static std::optional<PropagationEdge> GetNextPropagationEdge(const Graph& graph,
                                                             const PropagationEdge& edge) {
  if (edge.HasGraphOutput()) {
    return std::nullopt;
  }

  return GetNextPropagationEdge(graph, *edge.GetNodeAtEnd(graph, PropagationEdge::End::Destination));
}

Status PropagateDQForward(Graph& graph, bool& is_modified,
                          const std::unordered_set<std::string>& compatible_eps,
                          const logging::Logger& logger) {
  is_modified = false;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* dq_node_ptr = graph.GetNode(node_index);
    if (dq_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& dq_node = *dq_node_ptr;

    if (!IsSupportedDQNode(dq_node) ||
        !graph_utils::IsSupportedProvider(dq_node, compatible_eps) ||
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    std::vector<NodeArg*>& dq_input_defs = dq_node.MutableInputDefs();
    if (dq_input_defs.size() != QDQ::InputIndex::TOTAL_COUNT) {
      continue;
    }

    if (!optimizer_utils::IsScalar(*dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID]) ||
        !optimizer_utils::IsScalar(*dq_input_defs[QDQ::InputIndex::SCALE_ID])) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
        graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name());
    const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
        graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQ::InputIndex::SCALE_ID]->Name());

    if (nullptr == dq_zp_tensor_proto || nullptr == dq_scale_tensor_proto) {
      continue;
    }

    const auto edge_after_dq = GetNextPropagationEdge(graph, dq_node);
    if (!edge_after_dq) {
      continue;
    }

    for (auto curr_edge = GetNextPropagationEdge(graph, *edge_after_dq);
         curr_edge.has_value();
         curr_edge = GetNextPropagationEdge(graph, *curr_edge)) {
      if (const auto* src_node = curr_edge->GetNodeAtEnd(graph, PropagationEdge::End::Source);
          src_node && !CanNodePropagate(*src_node)) {
        break;
      }

      if (const auto* dst_node = curr_edge->GetNodeAtEnd(graph, PropagationEdge::End::Destination);
          dst_node && IsMatchingQDQPair(graph, *dst_node, dq_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge,
                                        *dq_input_defs[QDQ::InputIndex::SCALE_ID],
                                        *dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID],
                                        logger));
      is_modified = true;
    }

    //do {
    //  Node& next_node = *graph.GetNode(dq_node.OutputNodesBegin()->Index());
    //  if (!CanNodePropagate(next_node)) {
    //    // Try canceling out DQ/Q pair
    //    if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "QuantizeLinear", {10, 13}) &&
    //        graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders()) &&
    //        TryCancelOutDQQPair(graph, dq_node, next_node)) {
    //      is_modified = true;
    //    }

    //    break;
    //  }
    //  SwapAdjacentNodes(graph, dq_node, next_node);
    //  is_modified = true;
    //} while (optimizer_utils::CheckOutputEdges(graph, dq_node, 1));
  }

  return Status::OK();
}

Status PropagateQBackward(Graph& graph, bool& is_modified,
                          const std::unordered_set<std::string>& compatible_eps,
                          const logging::Logger& logger) {
  is_modified = false;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* q_node_ptr = graph.GetNode(node_index);
    if (q_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& q_node = *q_node_ptr;

    if (!IsSupportedQNode(q_node) ||
        !graph_utils::IsSupportedProvider(q_node, compatible_eps)) {
      continue;
    }

    std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();
    if (q_input_defs.size() != QDQ::InputIndex::TOTAL_COUNT ||
        !optimizer_utils::IsScalar(*q_input_defs[QDQ::InputIndex::ZERO_POINT_ID]) ||
        !optimizer_utils::IsScalar(*q_input_defs[QDQ::InputIndex::SCALE_ID])) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
        graph_utils::GetConstantInitializer(graph, q_input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name());
    const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
        graph_utils::GetConstantInitializer(graph, q_input_defs[QDQ::InputIndex::SCALE_ID]->Name());

    if (nullptr == q_zp_tensor_proto || nullptr == q_scale_tensor_proto) {
      continue;
    }

    const auto edge_before_q = GetPreviousPropagationEdge(graph, q_node);
    if (!edge_before_q) {
      continue;
    }

    for (auto curr_edge = GetPreviousPropagationEdge(graph, *edge_before_q);
         curr_edge.has_value();
         curr_edge = GetPreviousPropagationEdge(graph, *curr_edge)) {
      if (auto* dst_node = curr_edge->GetNodeAtEnd(graph, PropagationEdge::End::Destination);
          dst_node && !CanNodePropagate(*dst_node)) {
        break;
      }

      if (auto* src_node = curr_edge->GetNodeAtEnd(graph, PropagationEdge::End::Source);
          src_node && IsMatchingQDQPair(graph, q_node, *src_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge,
                                        *q_input_defs[QDQ::InputIndex::SCALE_ID],
                                        *q_input_defs[QDQ::InputIndex::ZERO_POINT_ID],
                                        logger));

      is_modified = true;
    }

    // ... -> Transpose -> MaxPool -> Q
    // ... ->QDQ-> Transpose ->QDQ-> MaxPool -> Q

    //do {
    //  if (q_node.InputNodesBegin() == q_node.InputNodesEnd()) {
    //    break;
    //  }

    //  Node& prev_node = *graph.GetNode(q_node.InputNodesBegin()->Index());
    //  if (!optimizer_utils::CheckOutputEdges(graph, prev_node, 1)) {
    //    break;
    //  }

    //  if (!CanNodePropagate(prev_node)) {
    //    // Try canceling out DQ/Q pair
    //    Node& dq_node = prev_node;
    //    if (graph_utils::IsSupportedOptypeVersionAndDomain(dq_node, "DequantizeLinear", {10, 13}) &&
    //        graph_utils::IsSupportedProvider(dq_node, GetCompatibleExecutionProviders()) &&
    //        TryCancelOutDQQPair(graph, dq_node, q_node)) {
    //      is_modified = true;
    //    }
    //    break;
    //  }

    //  SwapAdjacentNodes(graph, prev_node, q_node);
    //  is_modified = true;
    //} while (true);
  }

  return Status::OK();
}
}  // namespace

Status QDQPropagationTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));
  }

  while (true) {
    bool propagated_q = false, propagated_dq = false;
    ORT_RETURN_IF_ERROR(PropagateQBackward(graph, propagated_q, GetCompatibleExecutionProviders(), logger));
    ORT_RETURN_IF_ERROR(PropagateDQForward(graph, propagated_dq, GetCompatibleExecutionProviders(), logger));

    if (propagated_q || propagated_dq) {
      modified = true;
    } else {
      break;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
