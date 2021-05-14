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

/**
DynamicQuantizeMatMulFusion will fuse subgraph like below into DynamicQuantizeMatMul:
      (input)
         |
         v
  DynamicQuantizeLinear       B,B_Scale,B_Zero      Bias                          (input)                B,B_Scale,B_Zero      Bias
   |        |        |               |               |                              |                           |               |
   |        |        |               |               |                              |                           |               |
  A| A_Scale| A_Zero |               |               |                              |                           |               |
   |        |        |               |               |                              |                           |               |
   v        v        v               |               |                              |                           |               |
  MatMulIntegerToFloat <-------------+---------------+           ---->       DynamicQuantizeMatMul<-------------+---------------+
         |                                                                          |
         v                                                                          v
      (output)                                                                   (output)
 */
Status DynamicQuantizeMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& matmul_integer_to_float_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(matmul_integer_to_float_node, modified, graph_level, logger));

    auto& mtf_input_args = matmul_integer_to_float_node.MutableInputDefs();
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul_integer_to_float_node, "MatMulIntegerToFloat", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(matmul_integer_to_float_node, GetCompatibleExecutionProviders()) ||
        mtf_input_args.size() < 5 /*A zero point can not be optional*/) {
      continue;
    }

    const Node* p_dynamic_quant_linear = graph_utils::GetInputNode(matmul_integer_to_float_node, 0 /*arg_index*/);
    if (p_dynamic_quant_linear == nullptr) {
      continue;
    }

    Node& dynamic_quant_linear = *graph.GetNode(p_dynamic_quant_linear->Index());
    auto& dql_output_args = dynamic_quant_linear.MutableOutputDefs();
    if (!optimizer_utils::CheckOutputEdges(graph, dynamic_quant_linear, dql_output_args.size())) {
      continue;
    }

    auto& dql_Y_scale = dql_output_args[1];
    auto& dql_Y_zp = dql_output_args[2];
    auto& mtf_A_scale = mtf_input_args[2];
    auto& mtf_A_zp = mtf_input_args[4];
    if (dql_Y_scale != mtf_A_scale ||
        dql_Y_zp != mtf_A_zp) {
      continue;
    }

    NodeArg optional_node_arg("", nullptr);
    std::string op_type_to_fuse = "DynamicQuantizeMatMul";
    std::vector<NodeArg*> input_defs{
        dynamic_quant_linear.MutableInputDefs()[0],
        mtf_input_args[1],  // B of MatmulIntegerToFloat
        mtf_input_args[3],  // B_Scale of MatmulIntegerToFloat
        &optional_node_arg,
        &optional_node_arg};

    // B_ZeroPoint of MatmulIntegerToFloat
    if (mtf_input_args.size() >= 6) {
      input_defs[3] = mtf_input_args[5];

      // Bias of MatmulIntegerToFloat
      if (mtf_input_args.size() >= 7) {
        input_defs[4] = mtf_input_args[6];
      }
    }

    Node* fused_node = &graph.AddNode(matmul_integer_to_float_node.Name(),
                                      op_type_to_fuse,
                                      "",
                                      input_defs,
                                      matmul_integer_to_float_node.MutableOutputDefs(),
                                      nullptr,
                                      kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node->SetExecutionProviderType(matmul_integer_to_float_node.GetExecutionProviderType());

    nodes_to_remove.push_back(dynamic_quant_linear);
    nodes_to_remove.push_back(matmul_integer_to_float_node);
  }

  modified = !nodes_to_remove.empty();

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  return Status::OK();
}
}  // namespace onnxruntime
