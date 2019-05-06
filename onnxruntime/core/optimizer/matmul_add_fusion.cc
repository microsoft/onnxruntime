// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MatMulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {7}) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    Node& matmul_node = node;
    Node& add_node = const_cast<Node&>(next_node);
    std::vector<NodeArg> input_args, output_args;
    auto matmul_input_defs = matmul_node.MutableInputDefs();
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
    if (nullptr == matmul_a_shape || nullptr == matmul_b_shape) {
      continue;
    } else if (1 == matmul_a_shape->dim_size() && 2 == matmul_b_shape->dim_size()) {
      // MatMul has shape [K] * [K, N], reset it to [1, K] * [K, N], so that it can work for Gemm
      auto mutable_matmul_a_shape = const_cast<ONNX_NAMESPACE::TensorShapeProto*>(matmul_a_shape);
      auto dim_0 = mutable_matmul_a_shape->mutable_dim(0);
      auto dim_1 = (const_cast<ONNX_NAMESPACE::TensorShapeProto*>(matmul_a_shape))->add_dim();
      (*dim_1) = (*dim_0);
      dim_0->set_dim_value(1);
    }
    if (2 != matmul_a_shape->dim_size() || 2 != matmul_b_shape->dim_size()) {
      // Gemm only support Matrix
      continue;
    }

    auto matmul_output_name = matmul_node.OutputDefs()[0]->Name();
    auto gemm_input_defs = matmul_input_defs;
    if (matmul_output_name == add_input_defs[0]->Name()) {
      // matmul output as Add_A, should use Add_B as input C for gemm
      // Gemm only support unidirectional broadcast on C
      if (add_input_defs[1]->Shape()->dim_size() > 2) {
        continue;
      }
      gemm_input_defs.push_back(add_input_defs[1]);
    } else {
      // matmul output as Add_B, should use Add_A as input C for gemm
      // Gemm only support unidirectional broadcast on C
      if (add_input_defs[0]->Shape()->dim_size() > 2) {
        continue;
      }
      gemm_input_defs.push_back(add_input_defs[0]);
    }

    Node& gemm_node = graph.AddNode(graph.GenerateNodeName("gemm"),
                                    "Gemm",
                                    "fused Matmul and Add " + add_node.OpType(),
                                    gemm_input_defs,
                                    add_node.MutableOutputDefs());

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gemm_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());

    removed_nodes.push_front(matmul_node.Index());
    removed_nodes.push_front(add_node.Index());
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (auto it = removed_nodes.begin(); it != removed_nodes.end(); ++it) {
    graph.RemoveNode(*it);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
