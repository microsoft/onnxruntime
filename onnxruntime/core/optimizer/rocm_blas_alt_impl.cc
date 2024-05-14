// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "core/optimizer/initializer.h"
#include "core/optimizer/rocm_blas_alt_impl.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status RocmBlasAltImpl::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  bool is_backward_pass = false;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);

    if (node.OpType() == "YieldOp") {
      is_backward_pass = true;
    }

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (is_backward_pass) {
      node.AddAttribute(std::string("__backwardpass"), static_cast<int64_t>(1));
      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
