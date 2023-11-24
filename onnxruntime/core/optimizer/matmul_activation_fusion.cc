// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_activation_fusion.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace matmulactivationfusion_internal {
// Don't check if the op is Deprecated. In ONNX Runtime's world, there is no deprecation.
bool IsSupportedOptypeVersionAndDomain(const Node& node, const std::string& op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain) {
  return (node.OpType() == op_type && graph_utils::MatchesOpSinceVersion(node, versions) &&
          graph_utils::MatchesOpSetDomain(node, domain));
}

// If the op has multiple versions, here we require it must have a single implementation that can work across all the
// versions. Because in the fusion, we discarded the op version information.
bool IsFusableActivation(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Softmax", {1, 11, 13}, kOnnxDomain);
}
}  // namespace matmulactivationfusion_internal

Status MatMulActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedMatMul", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const Node& next_node = *(node.OutputNodesBegin());
    if (!matmulactivationfusion_internal::IsFusableActivation(next_node) || next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    Node& act_node = *graph.GetNode(next_node.Index());  // get mutable reference

    Node& fused_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_FusedActivation"), "FusedMatMulActivation",
                                     node.Description() + " with activation " + act_node.OpType(),
                                     node.MutableInputDefs(), {}, &node.GetAttributes(), kMSDomain);

    // Add a new attribute to specify the activation type
    fused_node.AddAttribute("activation", act_node.OpType());

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Add optional attributes for activations
    const NodeAttributes& attrs = act_node.GetAttributes();
    for (const auto& attr : attrs) {
      AttributeProto fused_node_attr(attr.second);
      fused_node_attr.set_name("activation_" + attr.first);
      fused_node.AddAttributeProto(std::move(fused_node_attr));
    }

    // move output definitions and edges from act_node to fused_node. delete node and act_node.
    graph_utils::FinalizeNodeFusion(graph, {node, act_node}, fused_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
