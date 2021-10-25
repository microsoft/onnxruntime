// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/api_impl.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

LayoutHandlerResult QLinearConvHandler(api::GraphRef& graph, api::NodeRef& node) {
  ORT_UNUSED_PARAMETER(graph);

  // Skip if domain is incorrect
  auto domain = node.Domain();
  if (domain != "" && domain != "com.microsoft") {
    return {false, 0, std::nullopt, std::nullopt};
  }

  // Skip if already transformed
  if (node.GetAttributeIntDefault("channels_last", 0) == 1) {
    return {false, 0, std::nullopt, std::nullopt};
  }

  // Skip if unknown rank
  auto shape = NodeFromApiNode(node).InputDefs()[0]->Shape();
  if (shape == nullptr) {
    return {false, 0, std::nullopt, std::nullopt};
  }

  // Convert to channels last
  size_t rank = shape->dim_size();
  node.SetAttributeInt("channels_last", 1);
  return {true, rank, std::nullopt, "com.microsoft"};
}

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  // Only QLinearConv needs to be handled explicitly. The rest will be transformed if needed during transpose
  // optimization.
  std::unordered_map<std::string_view, LayoutHandler> handler_map = {
      {"QLinearConv", &QLinearConvHandler}
  };

  auto api_graph = MakeApiGraph(graph, std::move(cpu_allocator_), logger, kCpuExecutionProvider);
  if (ChannelFirstToChannelLast(*api_graph, handler_map, /*allow_extended_ops*/ true)) {
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
