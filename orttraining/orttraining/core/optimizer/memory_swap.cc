// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/memory_swap.h"

#define PRINT_MEMSWAP_INFO

#ifdef PRINT_MEMSWAP_INFO
// for dumping the shape
namespace ONNX_NAMESPACE {
std::ostream& operator<<(std::ostream& out, const TensorShapeProto& shape_proto);
}
#endif

namespace onnxruntime {

Status MemorySwap::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  // Skip following ops that do not rely on tensor content (Shape),
  // and tensor being aliased (Flatten, Identity, Reshape, Squeeze, Unsqueeze),
  // and don't swap more than once (SwapFromHost, SwapToHost)
  static const std::unordered_set<std::string> ignore_op_types =
      {"Shape",
       "Flatten",
       "Identity",
       "Reshape",
       "Squeeze",
       "Unsqueeze",
       "SwapFromHost",
       "SwapToHost"};

  GraphViewer graph_viewer(graph);
  size_t topo_index = 0;
  std::unordered_map<NodeIndex, size_t> topo_indices;
  bool found_stop_at = false;
  for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    topo_indices.insert(std::make_pair(index, topo_index));
    ++topo_index;
    const Node* node = graph_viewer.GetNode(index);
    node->ForEachWithIndex(
        node->OutputDefs(),
        [&](const NodeArg& arg, size_t) {
          if (arg.Name() == stop_at_node_arg_)
            found_stop_at = true;
          return Status::OK();
        });
  }
  ORT_RETURN_IF_NOT(found_stop_at, "MemorySwap: could not find node with output arg: ", stop_at_node_arg_);

  for (const auto src_node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    auto is_backward = [topo_indices](NodeIndex i) -> bool {
      // only count dst node in BW. Here we assume the FW/BW nodes are roughly symmetric
      // note that we don't use node description because there might be fusion rules breaking that assumption
      return topo_indices.at(i) > topo_indices.size() / 2;
    };

    const Node& src_node = *(graph_viewer.GetNode(src_node_idx));

    bool done = false;
    src_node.ForEachWithIndex(
        src_node.OutputDefs(),
        [this, topo_index, &done](const NodeArg& arg, size_t) {
          if (arg.Name() == stop_at_node_arg_)
            done = true;
          return Status::OK();
        });

    if (done)
      break;

    // check if src_node should be handled
    if (ignore_op_types.count(src_node.OpType()) && !is_backward(src_node_idx))
      continue;

    // map from src_node_arg_idx to vector of pair(dst_node_idx, dst_node_arg_idx)
    std::unordered_map<int, std::vector<std::pair<NodeIndex, int>>> src_node_edges;
    for (auto edge_iter = src_node.OutputEdgesBegin(); edge_iter != src_node.OutputEdgesEnd(); ++edge_iter) {
      NodeIndex dst_node_idx = edge_iter->GetNode().Index();
      if (0 == ignore_op_types.count(edge_iter->GetNode().OpType()) && is_backward(dst_node_idx)) {
        auto src_node_arg_idx = edge_iter->GetSrcArgIndex();
        if (0 == src_node_edges.count(src_node_arg_idx)) {
          src_node_edges.insert(std::make_pair(src_node_arg_idx, std::vector<std::pair<NodeIndex, int>>()));
        }
        src_node_edges[src_node_arg_idx].push_back(std::make_pair(dst_node_idx, edge_iter->GetDstArgIndex()));
      }
    }

    for (auto edge_iter = src_node_edges.begin(); edge_iter != src_node_edges.end(); ++edge_iter) {
      auto src_node_arg_idx = edge_iter->first;
      const auto& dst_edge_ends = edge_iter->second;
      NodeArg* src_node_output_arg = const_cast<NodeArg*>(src_node.OutputDefs()[src_node_arg_idx]);
      auto& swap_out_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_out", src_node_output_arg->TypeAsProto());
      auto& swap_in_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_in", src_node_output_arg->TypeAsProto());
      auto& swap_out_node = graph.AddNode(src_node_output_arg->Name() + "_swapout",
                                          "SwapToHost",
                                          "",
                                          {src_node_output_arg},
                                          {&swap_out_arg});
      auto& swap_in_node = graph.AddNode(src_node_output_arg->Name() + "_swapin",
                                         "SwapFromHost",
                                         "Backward pass",
                                         {&swap_out_arg},
                                         {&swap_in_arg});
      // since memory swap transform is the last graph transformer, manually set their EP
      swap_out_node.SetExecutionProviderType(kCudaExecutionProvider);
      swap_in_node.SetExecutionProviderType(kCudaExecutionProvider);

#ifdef PRINT_MEMSWAP_INFO
      std::cout << "MemorySwap: " << src_node.Name() << "(" << src_node.OpType() << ") ";
      if (src_node_output_arg->Shape() != nullptr) {
        std::cout << *(src_node_output_arg->Shape());
      }
      std::cout << " ==> ";
#endif
      // process output edges from this output_def
      // note this needs to happen before linking src_node with swap_out_node
      // and since the operation might change src_node's OutputEdges, needs a copy of original edges
      for (const auto& dst_edge_end : dst_edge_ends) {
        NodeIndex dst_node_idx = dst_edge_end.first;
        int dst_arg_idx = dst_edge_end.second;
        // remove edge from src_node to dst_node
        graph.RemoveEdge(src_node.Index(), dst_node_idx, src_node_arg_idx, dst_arg_idx);
        // add edge from swap_in to dst_node
        graph.AddEdge(swap_in_node.Index(), dst_node_idx, 0, dst_arg_idx);
#ifdef PRINT_MEMSWAP_INFO
        std::cout << graph.GetNode(dst_node_idx)->Name() << "(" << graph.GetNode(dst_node_idx)->OpType() << "), ";
#endif
      }
#ifdef PRINT_MEMSWAP_INFO
      std::cout << std::endl;
#endif

      // add edges from src_node to swap_out, and swap_out to swap_in
      graph.AddEdge(src_node.Index(), swap_out_node.Index(), src_node_arg_idx, 0);
      graph.AddEdge(swap_out_node.Index(), swap_in_node.Index(), 0, 0);

      modified = true;
    }

    ++topo_index;
  }
  return Status::OK();
}

}  // namespace onnxruntime
