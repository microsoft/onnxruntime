// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "core/optimizer/initializer.h"
#include "core/optimizer/rocm_blas_alt_impl.h"
#include "core/graph/graph_utils.h"

#define PRE __FILE__ << ":" << __LINE__ << ":" << std::this_thread::get_id() << " "

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status RocmBlasAltImpl::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  bool is_backward_pass = false;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);

    std::cerr << PRE << node << std::endl;

#if 0
    if (node.OpType() == "YieldOp") {
      is_backward_pass = true;
      //std::cerr << PRE << "YieldOp found, before recurse" << std::endl;
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      //std::cerr << PRE << "YieldOp found, after recurse" << std::endl;
    }
    else
#else
    is_backward_pass = true;
#endif
    {
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    }

    //if (node.OpType() == "MatMul" || node.OpType() == "FusedMatMul" || node.OpType() == "Gemm") {
      //std::cerr << PRE << "HIT, is_backward_pass " << is_backward_pass << std::endl;
      if (is_backward_pass) {
        node.AddAttribute(std::string("__altimpl"), static_cast<int64_t>(1));
        modified = true;
      }
    //}
  }

  return Status::OK();
}
}  // namespace onnxruntime
