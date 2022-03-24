// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
#if defined(ORT_MINIMAL_BUILD)
  // update the producer/consumer info as previous optimizations may have invalidated it.
  // in a full build this will happen as part of Graph::Resolve.
  ORT_RETURN_IF_ERROR(graph.PopulateNodeArgToProducerConsumerLookupsFromNodes());
#endif

  GraphViewer graph_viewer(graph);
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  auto api_graph = MakeApiGraph(graph, cpu_allocator_, kCpuExecutionProvider);

  modified = false;
  for (std::unique_ptr<api::NodeRef>& node : api_graph->Nodes()) {
    // Only QLinearConv needs to be handled explicitly. The rest will be transformed if needed during transpose
    // optimization.
    if (node->OpType() == "QLinearConv") {
      auto domain = node->Domain();

      // Skip if domain is incorrect
      if (domain != kOnnxDomain && domain != kMSDomain) {
        continue;
      }

      // Skip if already transformed
      if (node->GetAttributeIntDefault("channels_last", 0) == 1) {
        continue;
      }

      // Skip if unknown rank
      auto shape = NodeFromApiNode(*node).InputDefs()[0]->Shape();
      if (shape == nullptr) {
        continue;
      }

      // Convert to channels last
      size_t rank = shape->dim_size();
      node->SetAttributeInt("channels_last", 1);

      std::vector<int64_t> input_perm = ChannelFirstToLastPerm(rank);
      std::vector<int64_t> output_perm = ChannelLastToFirstPerm(rank);
      WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});

      if (domain != kMSDomain) {
        SwapNodeOpTypeAndDomain(*api_graph, *node, "QLinearConv", kMSDomain);
      }

      modified = true;
    }
  }

  if (modified) {
    Optimize(*api_graph, /*allow_extended_ops*/ true);
  }

  return Status::OK();
}

}  // namespace onnxruntime
