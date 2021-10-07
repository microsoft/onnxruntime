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
namespace onnxruntime {

LayoutHandlerResult QLinearConvHandler(api::Graph& graph, api::Node& node) {
  ORT_UNUSED_PARAMETER(graph);
  if (node.Domain() != "") {
    return {false, 0, std::nullopt, std::nullopt};
  }
  if (node.GetAttributeIntDefault("channels_last", 0) == 1) {
    return {false, 0, std::nullopt, std::nullopt};
  }
  auto shape = static_cast<OrtNode&>(node).Node().InputDefs()[0]->Shape();
  if (shape == nullptr) {
    return {false, 0, std::nullopt, std::nullopt};
  }
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

  std::unordered_map<std::string_view, LayoutHandler*> handler_map = {
      {"QLinearConv", &QLinearConvHandler}
  };
  auto ort_graph = OrtGraph(graph, cpu_allocator_, logger, kCpuExecutionProvider);
  if (ChannelFirstToChannelLast(ort_graph, handler_map, /*allow_extended_ops*/ true)) {
    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
