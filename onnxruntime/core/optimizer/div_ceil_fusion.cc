// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/div_ceil_fusion.h"

#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

Status DivCeilFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Div", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *node->OutputNodesBegin();
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Ceil", {6, 13}) ||
        next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    // Only apply to float types.
    const auto* output_type = node->OutputDefs()[0]->TypeAsProto();
    if (output_type == nullptr || !output_type->has_tensor_type()) {
      continue;
    }
    auto elem_type = output_type->tensor_type().elem_type();
    if (elem_type != TensorProto_DataType_FLOAT && elem_type != TensorProto_DataType_FLOAT16) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(*node)) {
      continue;
    }

    auto& div_node = *node;
    auto& ceil_node = *graph.GetNode(next_node.Index());

    Node& div_ceil_node = graph.AddNode(graph.GenerateNodeName("DivCeil"),
                                        "DivCeil",
                                        "Fused Div+Ceil",
                                        div_node.MutableInputDefs(),
                                        {}, {}, kMSDomain);

    div_ceil_node.SetExecutionProviderType(div_node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, {div_node, ceil_node}, div_ceil_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
