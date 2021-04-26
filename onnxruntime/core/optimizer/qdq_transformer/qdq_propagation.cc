// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_propagation.h"

#include <deque>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

static constexpr size_t QDQInputCountRequired = 3;
static constexpr size_t QDQInputScaleIdx = 1;
static constexpr size_t QDQInputZeroPointIdex = 2;

static bool CanNodePropagate(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {5, 13}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13});
}

static bool TryCancelOutDQQPair(Graph& graph, Node& dq_node, Node& q_node) {
  std::vector<NodeArg*>& dq_input_defs = dq_node.MutableInputDefs();
  std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();
  if (dq_input_defs.size() != QDQInputCountRequired ||
      q_input_defs.size() != QDQInputCountRequired ||
      !graph.GetNodeOutputsInGraphOutputs(q_node).empty()) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto = nullptr;
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto = nullptr;
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto = nullptr;
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *dq_input_defs[QDQInputScaleIdx]) ||
      !graph_utils::NodeArgIsConstant(graph, *q_input_defs[QDQInputScaleIdx]) ||
      !graph_utils::NodeArgIsConstant(graph, *dq_input_defs[QDQInputZeroPointIdex]) ||
      !graph_utils::NodeArgIsConstant(graph, *q_input_defs[QDQInputZeroPointIdex]) ||
      !graph.GetInitializedTensor(dq_input_defs[QDQInputScaleIdx]->Name(), dq_scale_tensor_proto) ||
      !graph.GetInitializedTensor(q_input_defs[QDQInputScaleIdx]->Name(), q_scale_tensor_proto) ||
      !graph.GetInitializedTensor(dq_input_defs[QDQInputZeroPointIdex]->Name(), dq_zp_tensor_proto) ||
      !graph.GetInitializedTensor(q_input_defs[QDQInputZeroPointIdex]->Name(), q_zp_tensor_proto)) {
    return false;
  }

  Initializer q_zp(*q_zp_tensor_proto, graph.ModelPath());
  Initializer q_scale(*q_scale_tensor_proto, graph.ModelPath());
  Initializer dq_zp(*dq_zp_tensor_proto, graph.ModelPath());
  Initializer dq_scale(*dq_scale_tensor_proto, graph.ModelPath());

  if (*q_zp.data<int8_t>() != *dq_zp.data<int8_t>() ||
      *q_scale.data<float>() != *dq_scale.data<float>() ||
      dq_node.GetOutputEdgesCount() != 1) {
    return false;
  }

  // remove edge between parent of DQ to DQ
  std::pair<NodeIndex, int> input_edge_info{0, -1};
  auto* dq_input_edge_0 = graph_utils::GetInputEdge(dq_node, 0);
  if (dq_input_edge_0) {
    input_edge_info.first = dq_input_edge_0->GetNode().Index();
    input_edge_info.second = dq_input_edge_0->GetSrcArgIndex();
    graph.RemoveEdge(dq_input_edge_0->GetNode().Index(), dq_node.Index(), dq_input_edge_0->GetSrcArgIndex(), dq_input_edge_0->GetDstArgIndex());
  }

  graph_utils::RemoveNodeOutputEdges(graph, dq_node);  // Remove DQ node output edges

  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(q_node, 0);
  graph_utils::RemoveNodeOutputEdges(graph, q_node);  // Remove Q node output edges
  for (auto& output_edge : output_edges) {
    // set input NodeArg of Q's children to the 1st input of DQ
    graph.GetNode(output_edge.dst_node)->MutableInputDefs()[output_edge.dst_arg_index] = dq_input_defs[0];

    // add edge between parent of DQ to children of Q
    if (input_edge_info.second != -1) {
      graph.AddEdge(input_edge_info.first, output_edge.dst_node, input_edge_info.second, output_edge.dst_arg_index);
    }
  }

  graph.RemoveNode(dq_node.Index());
  graph.RemoveNode(q_node.Index());
  return true;
}

// A helper function that swap the relationship of 2 nodes.
// @param up_node The original parent node
// @param down_node The original child node
// after calling the function, the up_node will become the child of down_node
// Assumptions of this function are:
// 1. up_node only has one Edge that points to down_node and its output is not graph output.
// 2. NodeArg slots of the edge between up_node and down_node are (0, 0)
static void SwapAdjacentNodes(Graph& graph, Node& up_node, Node& down_node) {
  ORT_ENFORCE(optimizer_utils::CheckOutputEdges(graph, up_node, 1),
              "up_node should have only one Edge that points to down_node and its output is not graph output");

  auto edge_it = up_node.OutputEdgesBegin();
  ORT_ENFORCE(edge_it->GetDstArgIndex() == 0 &&
                  edge_it->GetSrcArgIndex() == 0 &&
                  edge_it->GetNode().Index() == down_node.Index(),
              "up_node should be parent of down_node and NodeArg slots of the edge between up_node and down_node should be (0, 0).");

  // ************** Remove edges **************
  // Remove the edge between parent of up_node and up_node, and keep the info of parent of up_node
  std::pair<NodeIndex, int> up_node_input_edge_info{0, -1};
  auto* up_node_input_edge_0 = graph_utils::GetInputEdge(up_node, 0);
  if (up_node_input_edge_0) {
    up_node_input_edge_info.first = up_node_input_edge_0->GetNode().Index();
    up_node_input_edge_info.second = up_node_input_edge_0->GetSrcArgIndex();
    graph.RemoveEdge(up_node_input_edge_0->GetNode().Index(),
                     up_node.Index(),
                     up_node_input_edge_0->GetSrcArgIndex(),
                     up_node_input_edge_0->GetDstArgIndex());
  }

  auto down_node_output_edges_info = graph_utils::GraphEdge::GetNodeOutputEdges(down_node);
  graph_utils::RemoveNodeOutputEdges(graph, up_node);
  graph_utils::RemoveNodeOutputEdges(graph, down_node);

  // *********** Rebuild NodeArg ****************/
  down_node.MutableInputDefs()[0] = up_node.MutableInputDefs()[0];
  up_node.MutableOutputDefs()[0] = down_node.MutableOutputDefs()[0];

  NodeArg* new_node_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SwapAdjacentNodes"), nullptr);
  down_node.MutableOutputDefs()[0] = new_node_arg;
  up_node.MutableInputDefs()[0] = new_node_arg;

  // *********** Rebuild Edges ****************/
  if (up_node_input_edge_info.second >= 0) {
    graph.AddEdge(up_node_input_edge_info.first, down_node.Index(), up_node_input_edge_info.second, 0);
  }

  graph.AddEdge(down_node.Index(), up_node.Index(), 0, 0);

  for (auto output_edge_info : down_node_output_edges_info) {
    graph.AddEdge(up_node.Index(), output_edge_info.dst_node, 0, output_edge_info.dst_arg_index);
  }
}

bool QDQPropagationTransformer::PropagateDQForward(Graph& graph) const {
  bool is_modified = false;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* dq_node_ptr = graph.GetNode(node_index);
    if (dq_node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& dq_node = *dq_node_ptr;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dq_node, "DequantizeLinear", {10, 13}) ||
        !graph_utils::IsSupportedProvider(dq_node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    std::vector<NodeArg*>& dq_input_defs = dq_node.MutableInputDefs();
    if (dq_input_defs.size() != QDQInputCountRequired) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto = nullptr;
    const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *dq_input_defs[QDQInputScaleIdx]) ||
        !graph_utils::NodeArgIsConstant(graph, *dq_input_defs[QDQInputZeroPointIdex]) ||
        !graph.GetInitializedTensor(dq_input_defs[QDQInputScaleIdx]->Name(), dq_scale_tensor_proto) ||
        !graph.GetInitializedTensor(dq_input_defs[QDQInputZeroPointIdex]->Name(), dq_zp_tensor_proto)) {
      continue;
    }

    do {
      Node& next_node = *graph.GetNode(dq_node.OutputNodesBegin()->Index());
      if (!CanNodePropagate(next_node)) {
        break;
      }
      SwapAdjacentNodes(graph, dq_node, next_node);
      is_modified = true;
    } while (optimizer_utils::CheckOutputEdges(graph, dq_node, 1));

    // Cancel out DQ/Q pair
    Node& q_node = *graph.GetNode(dq_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(q_node, "QuantizeLinear", {10, 13}) ||
        !graph_utils::IsSupportedProvider(q_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    is_modified = is_modified || TryCancelOutDQQPair(graph, dq_node, q_node);
  }

  return is_modified;
}

bool QDQPropagationTransformer::PropagateQBackward(Graph& graph) const {
  bool is_modified = false;

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* q_node_ptr = graph.GetNode(node_index);
    if (q_node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& q_node = *q_node_ptr;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(q_node, "QuantizeLinear", {10, 13}) ||
        !graph_utils::IsSupportedProvider(q_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();
    if (q_input_defs.size() != QDQInputCountRequired) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto = nullptr;
    const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[QDQInputScaleIdx]) ||
        !graph_utils::NodeArgIsConstant(graph, *q_input_defs[QDQInputZeroPointIdex]) ||
        !graph.GetInitializedTensor(q_input_defs[QDQInputScaleIdx]->Name(), q_scale_tensor_proto) ||
        !graph.GetInitializedTensor(q_input_defs[QDQInputZeroPointIdex]->Name(), q_zp_tensor_proto)) {
      continue;
    }

    do {
      if (q_node.InputNodesBegin() == q_node.InputNodesEnd()) {
        break;
      }
      Node& prev_node = *graph.GetNode(q_node.InputNodesBegin()->Index());
      if (!optimizer_utils::CheckOutputEdges(graph, prev_node, 1) ||
          !CanNodePropagate(prev_node)) {
        break;
      }

      SwapAdjacentNodes(graph, prev_node, q_node);
      is_modified = true;
    } while (true);

    // Cancel out DQ/Q pair
    if (q_node.InputNodesBegin() == q_node.InputNodesEnd()) {
      continue;
    }
    Node& dq_node = *graph.GetNode(q_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(dq_node, "DequantizeLinear", {10, 13}) ||
        !graph_utils::IsSupportedProvider(dq_node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    is_modified = is_modified || TryCancelOutDQQPair(graph, dq_node, q_node);
  }

  return is_modified;
}

Status QDQPropagationTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));
  }

  while (PropagateQBackward(graph) || PropagateDQForward(graph)) {
  }

  modified = true;

  return Status::OK();
}

}  // namespace onnxruntime
