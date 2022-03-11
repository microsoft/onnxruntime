// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_transpose_optimizer.h"
#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "optimizer_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

Status TransposeOptimizer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  auto api_graph = MakeApiGraph(graph, cpu_allocator_, /*new_node_ep*/ nullptr);
  OptimizeResult result = onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ false);
  if (result.error_msg) {
    // currently onnx_layout_transformation::Optimize only fails if we hit an unsupported opset.
    // we don't want to fail loading the model just because we can't optimize Transpose ops, so just log a warning
    LOGS(logger, WARNING) << "Transpose optimizer failed: " << result.error_msg.value();
  }

  if (result.graph_modified) {
    modified = true;
  }

  GraphViewer graph_viewer(graph);
  auto nodes = std::vector<std::unique_ptr<api::NodeRef>>();
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    ORT_RETURN_IF_ERROR(Recurse(*graph.GetNode(index), modified, graph_level, logger));
  }

  return Status::OK();
}

}  // namespace onnxruntime
