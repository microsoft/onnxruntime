// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_matmul_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

// Check if bias is a 1-D tensor, or N-D tensor with the prior N-1 dimension equal to 1.
// And its last dimension equal to MatMul's last dimension
static bool CheckBiasShape(const TensorShapeProto* bias_shape, const TensorShapeProto* matmul_shape) {
  if (nullptr == matmul_shape || matmul_shape->dim_size() <= 1 ||
      nullptr == bias_shape || bias_shape->dim_size() < 1) {
    return false;
  }

  // First N-1 dimension must equal to 1
  for (int i = 0; i < bias_shape->dim_size() - 1; i++) {
    if (bias_shape->dim(i).dim_value() != 1) {
      return false;
    }
  }

  int64_t bias_last_dim = bias_shape->dim(bias_shape->dim_size() - 1).dim_value();
  int64_t matmul_last_dim = matmul_shape->dim(matmul_shape->dim_size() - 1).dim_value();
  return bias_last_dim == matmul_last_dim && bias_last_dim > 0;
}

/**
DynamicQuantizeMatMulFusion will fuse subgraph like below into DynamicQuantizeMatMul:
    (input)
       |
       v
DynamicQuantizeLinear --------+
       |                      |
       v                      v
MatMulInteger (B const)      Mul (B const)                     (input)
       |                      |                                   |
       v                      v                                   v
     Cast ------------------>Mul             ---->       DynamicQuantizeMatMul
                              |                                   |
                              v                                   v
                             Add (B const, Optional)           (output)
                              |
                              v
                          (output)

It also fuses subgraph like below into MatMulIntegerToFloat:
                                              input                                                                            input
                                                |                                                                                |
                                                v                                                                                v
          +----------------------------DynamicQuantizeLinear------------------------+                                  DynamicQuantizeLinear
          |                                     |                                   |                                            |
          |                    +----------------+--------------+                    |                                  +---------+----------+
          |                    |                              |                     |                                  |                    |
          V                    v                              v                     v                                  V                    v
    MatMulInteger(B const)   Mul(B const)                MatMulInteger (B const)   Mul (B const)       --->   MatMulIntegerToFloat MatMulIntegerToFloat
          |                    |                               |                    |                                  |                    | 
          v                    v                               v                    v                                  v                    v 
        Cast ---------------->Mul                            Cast ---------------->Mul                              (output1) ----------(output2)
                               |                                                    |
                               v                                                    v
                              Add (B const, Optional)                              Add (B const, Optional)
                               |                                                    |
                               v                                                    v
                            (output1)                                            (output2)

 */
Status DynamicQuantizeMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& mul_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(mul_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        !graph_utils::IsSupportedProvider(mul_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Left Parents path
    std::vector<graph_utils::EdgeEndToMatch> left_parent_path{
        {0, 0, "Cast", {6, 9}, kOnnxDomain},
        {0, 0, "MatMulInteger", {10}, kOnnxDomain},
        {0, 0, "DynamicQuantizeLinear", {11}, kOnnxDomain}};

    std::vector<graph_utils::EdgeEndToMatch> right_parent_path{
        {0, 1, "Mul", {7}, kOnnxDomain},
        {1, 0, "DynamicQuantizeLinear", {11}, kOnnxDomain}};

    std::vector<std::reference_wrapper<Node>> left_nodes;
    std::vector<std::reference_wrapper<Node>> right_nodes;
    if (!graph_utils::FindPath(graph, mul_node, true, left_parent_path, left_nodes, logger) ||
        !graph_utils::FindPath(graph, mul_node, true, right_parent_path, right_nodes, logger)) {
      continue;
    }

    Node& cast_node = left_nodes[0];
    Node& matmulinteger_node = left_nodes[1];
    Node& dql_node_left = left_nodes[2];

    Node& mul_node_right = right_nodes[0];
    Node& dql_node_right = right_nodes[1];

    // Check if left DynamicQuantizeLinear is same as right DynamicQuantizeLinear
    if (dql_node_left.Index() != dql_node_right.Index()) {
      continue;
    }

    // Check Nodes' Edges count and Nodes' outputs are not in Graph output
    if (!optimizer_utils::CheckOutputEdges(graph, cast_node, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, matmulinteger_node, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node_right, 1)) {
      continue;
    }

    const NodeArg& matmulinteger_B = *(matmulinteger_node.InputDefs()[1]);
    if (!graph_utils::IsConstantInitializer(graph, matmulinteger_B.Name(), true)) {
      continue;
    }

    const NodeArg& mul_right_B = *(mul_node_right.InputDefs()[1]);
    if (!graph_utils::IsConstantInitializer(graph, mul_right_B.Name(), true)) {
      continue;
    }

    // Find bias node
    Node* add_node = nullptr;
    // const Node* add_node = FindBiasNode(graph, mul_node, ;
    if (optimizer_utils::CheckOutputEdges(graph, mul_node, 1)) {
      const Node* tmp_add_node = graph_utils::FirstChildByType(mul_node, "Add");
      if (nullptr != tmp_add_node) {
        const NodeArg& tmp_add_node_B = *(tmp_add_node->InputDefs()[1]);
        if (graph_utils::IsConstantInitializer(graph, tmp_add_node_B.Name(), true) &&
            CheckBiasShape(tmp_add_node_B.Shape(), matmulinteger_B.Shape())) {
          add_node = graph.GetNode(tmp_add_node->Index());
        }
      }
    }

    // DynamicQuantizeLinear outputs are only used by one MatMulInteger,
    // thus it can fused into DynamicQuantizeMatMul
    NodeArg optional_node_arg("", nullptr);
    std::vector<NodeArg*> input_defs;
    std::string op_type_to_fuse = "DynamicQuantizeMatMul";
    if (optimizer_utils::CheckOutputEdges(graph, dql_node_left, 3)) {
      input_defs.push_back(dql_node_left.MutableInputDefs()[0]);
      input_defs.push_back(matmulinteger_node.MutableInputDefs()[1]);
      input_defs.push_back(mul_node_right.MutableInputDefs()[1]);
      input_defs.push_back(&optional_node_arg);

      if (matmulinteger_node.InputDefs().size() == 4) {
        const NodeArg& matmulinteger_B_zp = *(matmulinteger_node.InputDefs()[3]);
        if (!graph_utils::IsConstantInitializer(graph, matmulinteger_B_zp.Name(), true)) {
          continue;
        }
        input_defs[3] = matmulinteger_node.MutableInputDefs()[3];
      }

      nodes_to_remove.push_back(dql_node_left);
    } else {
      op_type_to_fuse = "MatMulIntegerToFloat";

      input_defs.push_back(matmulinteger_node.MutableInputDefs()[0]);
      input_defs.push_back(matmulinteger_node.MutableInputDefs()[1]);
      input_defs.push_back(mul_node_right.MutableInputDefs()[0]);
      input_defs.push_back(mul_node_right.MutableInputDefs()[1]);
      input_defs.push_back(&optional_node_arg);
      input_defs.push_back(&optional_node_arg);

      if (matmulinteger_node.InputDefs().size() >= 3) {
        // Add zero point of A
        input_defs[4] = matmulinteger_node.MutableInputDefs()[2];

        // Add zero point of B
        if (matmulinteger_node.InputDefs().size() == 4) {
          const NodeArg& matmulinteger_B_zp = *(matmulinteger_node.InputDefs()[3]);
          if (!graph_utils::IsConstantInitializer(graph, matmulinteger_B_zp.Name(), true)) {
            continue;
          }
          input_defs[5] = matmulinteger_node.MutableInputDefs()[3];
        }
      }
    }

    if (add_node != nullptr) {
      input_defs.push_back(add_node->MutableInputDefs()[1]);
    }

    Node* fused_node = &graph.AddNode(graph.GenerateNodeName(op_type_to_fuse),
                                      op_type_to_fuse,
                                      "",
                                      input_defs,
                                      add_node != nullptr ? add_node->MutableOutputDefs() : mul_node.MutableOutputDefs(),
                                      nullptr,
                                      kMSDomain);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    ORT_ENFORCE(nullptr != fused_node);
    fused_node->SetExecutionProviderType(mul_node.GetExecutionProviderType());

    nodes_to_remove.push_back(matmulinteger_node);
    nodes_to_remove.push_back(cast_node);
    nodes_to_remove.push_back(mul_node_right);
    nodes_to_remove.push_back(mul_node);
    if (add_node != nullptr) {
      nodes_to_remove.push_back(*add_node);
    }
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
