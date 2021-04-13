// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/isinf_reducesum_fusion.h"

#include "onnx/defs/attr_proto_util.h"

#include "core/optimizer/initializer.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
/*
This transform changes the following subgraph pattern:
Cast --> IsInf --> Cast --> ReduceSum --> Greater
to
IsAllFinite --> Not
*/
Status IsInfReduceSumFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr)
      continue;  // node was removed

    auto& isinf_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(isinf_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(isinf_node, "IsInf", {10}) ||
        isinf_node.GetOutputEdgesCount() != 1 ||
        !graph.GetNodeOutputsInGraphOutputs(isinf_node).empty()) {
      continue;
    }

    std::vector<NodeArg*> input_defs = isinf_node.MutableInputDefs();
    // see if there is a Cast before IsInf
    // This will happen if input type is FP16 but IsInf doesnt support fp16, so it will be cast to float/double
    // This Cast can be skipped as we are replacing the subgraph with IsAllFinite, which supports FP16
    auto cast1_node_iter = isinf_node.InputNodesBegin();
    if (cast1_node_iter != isinf_node.InputNodesEnd() &&
        graph_utils::IsSupportedOptypeVersionAndDomain(*cast1_node_iter, "Cast", {9, 13}) &&
        cast1_node_iter->GetOutputEdgesCount() == 1) {
      // check input type of cast node
      Node& cast1_node = *graph.GetNode(cast1_node_iter->Index());
      auto cast1_input_defs = cast1_node.MutableInputDefs();
      auto cast1_input_type = cast1_input_defs[0] ? cast1_input_defs[0]->Type() : nullptr;
      // remove cast only if the input type is float16
      if (cast1_input_type != nullptr && (*cast1_input_type) == "tensor(float16)") {
        input_defs = cast1_input_defs;
        nodes_to_remove.push_back(cast1_node);
      }
    }
    nodes_to_remove.push_back(isinf_node);

    auto cast2_node_itr = isinf_node.OutputNodesBegin();
    if (cast2_node_itr == isinf_node.OutputNodesEnd()) {
      continue;
    }

    Node& cast2_node = *graph.GetNode(cast2_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast2_node, "Cast", {9, 13}) ||
        cast2_node.GetOutputEdgesCount() != 1 ||
        !graph.GetNodeOutputsInGraphOutputs(cast2_node).empty()) {
      continue;
    }
    nodes_to_remove.push_back(cast2_node);

    auto reduce_sum_node_itr = cast2_node.OutputNodesBegin();
    if (reduce_sum_node_itr == cast2_node.OutputNodesEnd()) {
      continue;
    }

    Node& reduce_sum_node = *graph.GetNode(reduce_sum_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_sum_node, "ReduceSum", {1, 11, 13}) ||
        reduce_sum_node.GetOutputEdgesCount() != 1 ||
        !graph.GetNodeOutputsInGraphOutputs(reduce_sum_node).empty()) {
      continue;
    }
    nodes_to_remove.push_back(reduce_sum_node);

    auto greater_node_itr = reduce_sum_node.OutputNodesBegin();
    if (greater_node_itr == reduce_sum_node.OutputNodesEnd()) {
      continue;
    }

    Node& greater_node = *graph.GetNode(greater_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(greater_node, "Greater", {1, 7, 9, 13})) {
      continue;
    }
    nodes_to_remove.push_back(greater_node);

    Node& isallfinite_node = graph.AddNode(graph.GenerateNodeName("IsAllFinite"),
                                           "IsAllFinite",
                                           "fused " + isinf_node.Name(),
                                           input_defs,
                                           {},
                                           {},
                                           kMSDomain);
    isallfinite_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("is_all_finite"), nullptr));
    isallfinite_node.AddAttribute("isinf_only", static_cast<int64_t>(1));

    Node& not_node = graph.AddNode(graph.GenerateNodeName("Not"),
                                   "Not",
                                   "not of " + isallfinite_node.Name(),
                                   isallfinite_node.MutableOutputDefs(),
                                   {});
    // Add edge between newly added nodes
    graph.AddEdge(isallfinite_node.Index(), not_node.Index(), 0, 0);

    // Assign provider to the new nodes.
    isallfinite_node.SetExecutionProviderType(reduce_sum_node.GetExecutionProviderType());
    not_node.SetExecutionProviderType(reduce_sum_node.GetExecutionProviderType());

    // move output definitions and edge, remove nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, isallfinite_node, not_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
