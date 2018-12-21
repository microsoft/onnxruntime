// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/initializer.h"
#include "core/graph/matmul_add_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MatMulAddFusion::Apply(Graph& graph, bool& modified) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  for (auto node_index : node_topology_list) {
    auto node = graph.GetNode(node_index);
    if (nullptr == node ||
        !utils::IsSupportedOptypeVersionAndDomain(*node, "MatMul", 9) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    auto next_node_itr = node->OutputNodesBegin();
    if (next_node_itr == node->OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", 7)) {
      continue;
    }

    Node* matmul_node = node;
    Node& add_node = const_cast<Node&>(next_node);
    std::vector<NodeArg> input_args, output_args;
    auto matmul_input_defs = matmul_node->MutableInputDefs();
    auto add_input_defs = add_node.MutableInputDefs();

    // Gemm only support float, so the inputs of MatMul
    auto matmul_type = matmul_input_defs[0]->Type();
    auto add_type = add_input_defs[0]->Type();
    if ((*matmul_type) != "tensor(float)" || (*add_type) != "tensor(float)") {
      continue;
    }

    // Gemm only support Matrix, need to check the shape of MatMul and Add
    auto matmul_a_shape = matmul_input_defs[0]->Shape();
    auto matmul_b_shape = matmul_input_defs[1]->Shape();
    if (nullptr == matmul_a_shape || nullptr == matmul_b_shape ||
        2 != matmul_a_shape->dim_size() || 2 != matmul_b_shape->dim_size()) {
      continue;
    }

    auto matmul_output_name = matmul_node->OutputDefs()[0]->Name();
    auto gemm_input_defs = matmul_input_defs;
    if (matmul_output_name == add_input_defs[0]->Name()) {
      // matmul output as Add_A, should used Add_B as input C for gemm
      // Gemm only support unidirectional broadcast on C
      if (add_input_defs[1]->Shape()->dim_size() > 2) {
        continue;
      }
      gemm_input_defs.push_back(add_input_defs[1]);
    } else {
      // matmul output as Add_B, should used Add_A as input C for gemm
      // Gemm only support unidirectional broadcast on C
      if (add_input_defs[0]->Shape()->dim_size() > 2) {
        continue;
      }
      gemm_input_defs.push_back(add_input_defs[0]);
    }

    graph.AddNode(graph.GenerateNodeName("gemm"),
                  "Gemm",
                  "fused Matmul and Add " + add_node.OpType(),
                  gemm_input_defs,
                  add_node.MutableOutputDefs());

    removed_nodes.push_front(matmul_node->Index());
    removed_nodes.push_front(add_node.Index());
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (auto it = removed_nodes.begin(); it != removed_nodes.end(); ++it) {
    graph.RemoveNode(*it);
  }

  if (!removed_nodes.empty()) {
    modified = true;
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }

  return Status::OK();
}
}  // namespace onnxruntime
