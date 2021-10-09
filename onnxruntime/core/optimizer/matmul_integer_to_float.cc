// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer_to_float.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

// Check if bias is a 1-D tensor, or N-D tensor with the prior N-1 dimension equal to 1.
// And its last dimension equal to MatMul's last dimension
static bool CheckBiasShape(const TensorShapeProto* bias_shape) {
  if (nullptr == bias_shape || bias_shape->dim_size() < 1) {
    return false;
  }

  // First N-1 dimension must equal to 1
  for (int i = 0; i < bias_shape->dim_size() - 1; i++) {
    if (bias_shape->dim(i).dim_value() != 1) {
      return false;
    }
  }

  int64_t bias_last_dim = bias_shape->dim(bias_shape->dim_size() - 1).dim_value();

  // Don't allow last dimension to be 1, to be on the safe side
  return bias_last_dim > 1;
}

/**
MatMulIntegerToFloatFusion will fuse subgraph like below into MatMulIntegerToFloat:
 
 A   A_Zero B B_Zero  A_Scale) B_Scale  Bias (Const, Optional)
  \    |    |    /        \      /             |
   \   |    |   /          \    /              |
    \  |    |  /            \  /               |  
    MatMulInteger            Mul               |                             (A, B, A_Scale, B_Scale, A_Zero, B_Zero, Bias)
      |                       |                |                                               |
      v                       v                |                                               v
     Cast ------------------>Mul               |          ---->                       MatMulIntegerToFloat
                              |                |                                               |
                              v                |                                               v
                             Add <-------------+                                           (output)
                              |
                              v
                          (output)

 */
Status MatMulIntegerToFloatFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& mul_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(mul_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(mul_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    const Node* p_cast_node = graph_utils::FirstParentByType(mul_node, "Cast");
    if (p_cast_node == nullptr) {
      continue;
    }

    const Node* p_matmulinteger_node = graph_utils::FirstParentByType(*p_cast_node, "MatMulInteger");
    if (p_matmulinteger_node == nullptr) {
      continue;
    }

    const Node* p_mul_node_right = graph_utils::FirstParentByType(mul_node, "Mul");
    if (p_mul_node_right == nullptr) {
      continue;
    }

    Node& cast_node = *graph.GetNode(p_cast_node->Index());
    Node& matmulinteger_node = *graph.GetNode(p_matmulinteger_node->Index());
    Node& mul_node_right = *graph.GetNode(p_mul_node_right->Index());

    // Check Nodes' Edges count and Nodes' outputs are not in Graph output
    if (!optimizer_utils::CheckOutputEdges(graph, cast_node, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, matmulinteger_node, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node_right, 1)) {
      continue;
    }

    // Find bias node
    Node* p_add_node = nullptr;
    if (optimizer_utils::CheckOutputEdges(graph, mul_node, 1)) {
      const Node* tmp_add_node = graph_utils::FirstChildByType(mul_node, "Add");
      if (nullptr != tmp_add_node) {
        const NodeArg& tmp_add_node_B = *(tmp_add_node->InputDefs()[1]);
        if (graph_utils::IsConstantInitializer(graph, tmp_add_node_B.Name(), true) &&
            CheckBiasShape(tmp_add_node_B.Shape())) {
          p_add_node = graph.GetNode(tmp_add_node->Index());
        }
      }
    }

    // DynamicQuantizeLinear outputs are only used by one MatMulInteger,
    // thus it can fused into DynamicQuantizeMatMul
    NodeArg optional_node_arg("", nullptr);
    std::vector<NodeArg*> input_defs{
        matmulinteger_node.MutableInputDefs()[0],
        matmulinteger_node.MutableInputDefs()[1],
        mul_node_right.MutableInputDefs()[0],
        mul_node_right.MutableInputDefs()[1],
        &optional_node_arg,
        &optional_node_arg};

    if (p_matmulinteger_node->InputDefs().size() >= 3) {
      // Add zero point of A
      input_defs[4] = matmulinteger_node.MutableInputDefs()[2];

      // Add zero point of B
      if (p_matmulinteger_node->InputDefs().size() >= 4) {
        input_defs[5] = matmulinteger_node.MutableInputDefs()[3];
      }
    }

    if (p_add_node != nullptr) {
      input_defs.push_back(p_add_node->MutableInputDefs()[1]);
    }

    std::string op_type = "MatMulIntegerToFloat";
    Node& fused_node = graph.AddNode(matmulinteger_node.Name(),
                                     op_type,
                                     "",
                                     input_defs,
                                     p_add_node != nullptr ? p_add_node->MutableOutputDefs() : mul_node.MutableOutputDefs(),
                                     nullptr,
                                     kMSDomain);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(mul_node.GetExecutionProviderType());

    nodes_to_remove.push_back(matmulinteger_node);
    nodes_to_remove.push_back(cast_node);
    nodes_to_remove.push_back(mul_node_right);
    nodes_to_remove.push_back(mul_node);
    if (p_add_node != nullptr) {
      nodes_to_remove.push_back(*p_add_node);
    }
  }

  modified = modified || !nodes_to_remove.empty();

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  return Status::OK();
}
}  // namespace onnxruntime
