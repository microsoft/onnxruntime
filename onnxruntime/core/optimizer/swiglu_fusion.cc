// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/swiglu_fusion.h"

#include <array>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// QuickGelu's schema default; QuickGeluFusion always writes the attribute explicitly.
constexpr float kQuickGeluDefaultAlpha = 1.702f;

// Mul broadcasts but SwiGLU does not, so both operands must provably share one shape.
bool HaveSameShape(const NodeArg& a, const NodeArg& b) {
  const auto* shape_a = a.Shape();
  const auto* shape_b = b.Shape();
  if (shape_a == nullptr || shape_b == nullptr || shape_a->dim_size() == 0 ||
      shape_a->dim_size() != shape_b->dim_size()) {
    return false;
  }

  for (int i = 0; i < shape_a->dim_size(); ++i) {
    const auto& dim_a = shape_a->dim(i);
    const auto& dim_b = shape_b->dim(i);
    if (utils::HasDimValue(dim_a) && utils::HasDimValue(dim_b)) {
      if (dim_a.dim_value() != dim_b.dim_value()) return false;
    } else if (utils::HasDimParam(dim_a) && utils::HasDimParam(dim_b)) {
      if (dim_a.dim_param() != dim_b.dim_param()) return false;
    } else {
      return false;
    }
  }
  return true;
}

bool IsSupportedType(const NodeArg& arg) {
  const auto* type = arg.Type();
  return type != nullptr &&
         (*type == "tensor(float)" || *type == "tensor(float16)" || *type == "tensor(bfloat16)");
}

}  // namespace

Status SwiGluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr) continue;

    Node& mul_node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(mul_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(mul_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    int gate_index = -1;
    for (int i = 0; i < 2; ++i) {
      const Node* input_node = graph_utils::GetInputNode(mul_node, i);
      if (input_node != nullptr &&
          graph_utils::IsSupportedOptypeVersionAndDomain(*input_node, "QuickGelu", {1}, kMSDomain) &&
          input_node->GetExecutionProviderType() == mul_node.GetExecutionProviderType() &&
          optimizer_utils::CheckOutputEdges(graph, *input_node, 1)) {
        gate_index = i;
        break;
      }
    }
    if (gate_index < 0) continue;

    const int up_index = 1 - gate_index;
    Node& quick_gelu_node = *graph.GetNode(graph_utils::GetInputNode(mul_node, gate_index)->Index());
    NodeArg* gate_arg = quick_gelu_node.MutableInputDefs()[0];
    NodeArg* up_arg = mul_node.MutableInputDefs()[up_index];
    if (!IsSupportedType(*gate_arg) || !HaveSameShape(*gate_arg, *up_arg)) continue;

    const auto* alpha_attr = graph_utils::GetNodeAttribute(quick_gelu_node, "alpha");
    const float alpha = alpha_attr != nullptr ? alpha_attr->f() : kQuickGeluDefaultAlpha;

    // FinalizeNodeFusion only moves input edges of the first fused node, so `up` is re-wired below.
    const Node::EdgeEnd* up_edge = graph_utils::GetInputEdge(mul_node, up_index);
    const NodeIndex up_src_node = up_edge != nullptr ? up_edge->GetNode().Index() : 0;
    const int up_src_arg_index = up_edge != nullptr ? up_edge->GetSrcArgIndex() : -1;

    Node& swiglu_node = graph.AddNode(graph.GenerateNodeName(mul_node.Name() + "/SwiGluFusion/"), "SwiGLU",
                                      "fused QuickGelu and Mul", std::array{gate_arg, up_arg},
                                      std::array{mul_node.MutableOutputDefs()[0]}, nullptr, kMSDomain);
    swiglu_node.AddAttribute("activation_alpha", alpha);
    swiglu_node.SetExecutionProviderType(mul_node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, {quick_gelu_node, mul_node}, swiglu_node);
    if (up_src_arg_index >= 0) {
      graph.AddEdge(up_src_node, swiglu_node.Index(), up_src_arg_index, 1);
    }
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
