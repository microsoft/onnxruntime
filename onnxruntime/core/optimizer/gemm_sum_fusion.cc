// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gemm_sum_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status GemmSumFusion::Apply(Graph& graph, Node& gemm_node, RewriteRuleEffect& modified, const logging::Logger&) const {
  // Get currently set attributes of Gemm. Beta will become 1.0.
  bool transA = static_cast<bool>(gemm_node.GetAttributes().at("transA").i());
  bool transB = static_cast<bool>(gemm_node.GetAttributes().at("transB").i());
  float alpha = gemm_node.GetAttributes().at("alpha").f();
  float beta = 1.0f;

  Node& sum_node = *graph.GetNode(gemm_node.OutputEdgesBegin()->GetNode().Index());

  // Our first two new gemm's input defs are exactly the same as the old gemm's
  // output defs.
  std::vector<NodeArg*> new_gemm_input_defs = gemm_node.MutableInputDefs();

  // The other new gemm's input def is the other sum input def.
  // Also (maybe) get the sum node's other input edge, unless it's a graph input.
  if (sum_node.InputDefs()[0]->Name() == gemm_node.OutputDefs()[0]->Name()) {
    new_gemm_input_defs.push_back(sum_node.MutableInputDefs()[1]);
  } else {
    new_gemm_input_defs.push_back(sum_node.MutableInputDefs()[0]);
  }
  ORT_ENFORCE(new_gemm_input_defs.size() == 3);

  std::vector<NodeArg*> new_gemm_output_defs = sum_node.MutableOutputDefs();
  ORT_ENFORCE(new_gemm_output_defs.size() == 1);

  Node& new_gemm_node = graph.AddNode(graph.GenerateNodeName(gemm_node.Name() + "_sum_transformed"),
                                      gemm_node.OpType(),
                                      "Fused Gemm with Sum",
                                      new_gemm_input_defs,
                                      new_gemm_output_defs,
                                      {},
                                      gemm_node.Domain());
  new_gemm_node.AddAttribute("transA", static_cast<int64_t>(transA));
  new_gemm_node.AddAttribute("transB", static_cast<int64_t>(transB));
  new_gemm_node.AddAttribute("alpha", static_cast<float>(alpha));
  new_gemm_node.AddAttribute("beta", static_cast<float>(beta));

  // Move both input edges from original gemm to new gemm.
  for (auto input_edge : graph_utils::GraphEdge::GetNodeInputEdges(gemm_node)) {
    ORT_ENFORCE(input_edge.src_arg_index < 2);
    graph.AddEdge(input_edge.src_node, new_gemm_node.Index(), input_edge.src_arg_index, input_edge.dst_arg_index);
    graph.RemoveEdge(input_edge.src_node, input_edge.dst_node, input_edge.src_arg_index, input_edge.dst_arg_index);
  }

  // Move all output edges from sum to new gemm.
  for (auto output_edge : graph_utils::GraphEdge::GetNodeOutputEdges(sum_node)) {
    ORT_ENFORCE(output_edge.src_arg_index == 0);
    graph.AddEdge(new_gemm_node.Index(), output_edge.dst_node, output_edge.src_arg_index, output_edge.dst_arg_index);
    graph.RemoveEdge(output_edge.src_node, output_edge.dst_node, output_edge.src_arg_index, output_edge.dst_arg_index);
  }

  // Finally, move the other sum input edge to "C" for the new gemm node.
  // If The other sum input def is a a graph input, there is no edge to move.
  bool sum_input_moved = false;
  for (auto sum_input : graph_utils::GraphEdge::GetNodeInputEdges(sum_node)) {
    if (sum_input.src_node != gemm_node.Index()) {
      ORT_ENFORCE(!sum_input_moved);
      graph.AddEdge(sum_input.src_node, new_gemm_node.Index(), sum_input.src_arg_index, 2);
      graph.RemoveEdge(sum_input.src_node, sum_input.dst_node, sum_input.src_arg_index, sum_input.dst_arg_index);
      sum_input_moved = true;
    }
  }
  // Old gemm node output is no longer needed. It was previously fed into the
  // sum node which is now also handled by the new gemm. Remove this output edge
  // to allow the node to be removed from the graph.
  graph_utils::RemoveNodeOutputEdges(graph, gemm_node);

  ORT_ENFORCE(graph.RemoveNode(gemm_node.Index()));
  ORT_ENFORCE(graph.RemoveNode(sum_node.Index()));

  modified = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool GemmSumFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Perform a series of checks. If any fail, fusion may not be performed.

  // C must be missing for this fusion pattern to be valid.
  // Supported for Opset >=11 as earlier opsets have C as a required input
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {11, 13}) ||
      graph.NodeProducesGraphOutput(node) ||
      // Verify that C is missing. 2 inputs means that A and B are present, but not C.
      node.InputDefs().size() != 2) {
    return false;
  }

  // This gemm node must have exactly one output for this fusion pattern to be valid.
  if (node.OutputDefs().size() != 1) {
    return false;
  }

  const NodeArg* node_output = node.OutputDefs()[0];
  const Node& output_node = node.OutputEdgesBegin()->GetNode();

  // Fusion can be applied if the only output node is a Sum with exactly two inputs.
  if (
      !graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Sum", {1, 6, 8, 13}) ||
      output_node.InputDefs().size() != 2 ||
      // Make sure the two nodes do not span execution providers.
      output_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  // Sum must have the same input types, data types do not need to be checked.

  // Get the other sum input.
  const NodeArg* other_sum_input = nullptr;
  if (output_node.InputDefs()[0]->Name() == node_output->Name()) {
    other_sum_input = output_node.InputDefs()[1];
  } else {
    other_sum_input = output_node.InputDefs()[0];
  }
  ORT_ENFORCE(other_sum_input != nullptr);

  // valid bias_shapes are (N) or (1, N) or (M, 1) or (M, N) as
  // GEMM only supports unidirectional broadcast on the bias input C
  if (!other_sum_input->Shape()) {
    return false;
  }

  if (!node_output->Shape() || node_output->Shape()->dim_size() != 2) {
    return false;
  }

  const auto& bias_shape = *other_sum_input->Shape();
  const auto& matmul_output_shape = *node_output->Shape();
  const auto& M = matmul_output_shape.dim()[0];
  const auto& N = matmul_output_shape.dim()[1];
  auto dim_has_value_1 = [](const TensorShapeProto_Dimension& dim) {
    return dim.has_dim_value() && dim.dim_value() == 1;
  };

  // TODO: This is based on matmul_add_fusion.cc. Should this support the case where bias_shape is a scalar?
  bool valid = ((bias_shape.dim_size() == 1 && bias_shape.dim()[0] == N) ||
                (bias_shape.dim_size() == 2 && dim_has_value_1(bias_shape.dim()[0]) && bias_shape.dim()[1] == N) ||
                (bias_shape.dim_size() == 2 && bias_shape.dim()[0] == M &&
                 (dim_has_value_1(bias_shape.dim()[1]) || bias_shape.dim()[1] == N)));
  if (!valid) {
    return false;
  }

  // If none of the above checks specify render this fusion invalid, fusion is valid.
  return true;
}

}  // namespace onnxruntime
