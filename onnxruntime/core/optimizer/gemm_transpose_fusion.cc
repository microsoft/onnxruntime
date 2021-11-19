// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gemm_transpose_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status GemmTransposeFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& modified, const logging::Logger&) const {
  auto& gemm_node = node;
  const Node* A_node_ptr = graph_utils::GetInputNode(gemm_node, 0);
  const Node* B_node_ptr = graph_utils::GetInputNode(gemm_node, 1);
  const auto output_node_ptr = gemm_node.OutputNodesBegin();

  // get currently set attributes of Gemm
  bool transA = static_cast<bool>(gemm_node.GetAttributes().at("transA").i());
  bool transB = static_cast<bool>(gemm_node.GetAttributes().at("transB").i());

  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  auto new_gemm_input_defs = gemm_node.MutableInputDefs();

  // check if input A is a Transpose
  if (A_node_ptr != nullptr && A_node_ptr->OpType() == "Transpose") {
    // make sure all consumers are gemm nodes to avoid possible double transpose 
    std::vector<const Node*> gemm_nodes = graph_utils::FindChildrenByType(*A_node_ptr, "Gemm");
    if (gemm_nodes.size() == A_node_ptr->GetOutputEdgesCount()) {
      Node& A_node = *graph.GetNode(A_node_ptr->Index());
      transA = !transA;
      if (A_node.GetOutputEdgesCount() > 1) {
        // remove only the edge between the Transpose and Gemm nodes, the Transpose won't be removed 
        // since it's still connected to other Gemm. When transformation for the last connected Gemm is
        // being processed, it would fall into the else {} below to remove the Transpose node 
        int output_idx = graph_utils::GetNodeOutputIndexFromOutputName(A_node, gemm_node.MutableInputDefs()[0]->Name());
        graph.RemoveEdge(A_node.Index(), gemm_node.Index(), output_idx, 0);
      } else {
        nodes_to_remove.push_back(A_node);
      }
      new_gemm_input_defs[0] = A_node.MutableInputDefs()[0];
    }
  }
  // check if input B is a Transpose
  if (B_node_ptr != nullptr && B_node_ptr->OpType() == "Transpose") {
    std::vector<const Node*> gemm_nodes = graph_utils::FindChildrenByType(*B_node_ptr, "Gemm");
    if (gemm_nodes.size() == B_node_ptr->GetOutputEdgesCount()) {
      Node& B_node = *graph.GetNode(B_node_ptr->Index());
      transB = !transB;
      if (B_node.GetOutputEdgesCount() > 1) {
        int output_idx = graph_utils::GetNodeOutputIndexFromOutputName(B_node, gemm_node.MutableInputDefs()[1]->Name());
        graph.RemoveEdge(B_node.Index(), gemm_node.Index(), output_idx, 1);
      } else {
        nodes_to_remove.push_back(B_node);
      }
      new_gemm_input_defs[1] = B_node.MutableInputDefs()[0];
    }
  }

  nodes_to_remove.push_back(gemm_node);

  // check if output node is Transpose
  if (output_node_ptr != gemm_node.OutputNodesEnd() &&
      gemm_node.InputDefs().size() <= 2 &&  // C is missing
      output_node_ptr->OpType() == "Transpose") {
    Node& output_node = *graph.GetNode(output_node_ptr->Index());
    // (AB)' = B'A' : reverse the inputs
    std::reverse(new_gemm_input_defs.begin(), new_gemm_input_defs.end());
    bool new_transB = !transA;
    transA = !transB;
    transB = new_transB;

    nodes_to_remove.push_back(output_node);
  }

  Node& new_gemm_node = graph.AddNode(graph.GenerateNodeName(gemm_node.Name() + "_transformed"),
                                      gemm_node.OpType(),
                                      "Fused Gemm with Transpose",
                                      new_gemm_input_defs,
                                      {},
                                      {},
                                      gemm_node.Domain());
  new_gemm_node.AddAttribute("transA", static_cast<int64_t>(transA));
  new_gemm_node.AddAttribute("transB", static_cast<int64_t>(transB));
  new_gemm_node.AddAttribute("alpha", gemm_node.GetAttributes().at("alpha").f());
  new_gemm_node.AddAttribute("beta", gemm_node.GetAttributes().at("beta").f());

  graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, new_gemm_node);

  modified = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool GemmTransposeFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {1, 6, 7, 9, 11, 13}) ||
      node.GetOutputEdgesCount() > 1) {
    return false;
  }

  // Fusion can be applied if there is a transpose at either of the inputs
  for (auto node_it = node.InputNodesBegin(); node_it != node.InputNodesEnd(); ++node_it) {
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*node_it, "Transpose", {1, 13}) &&
        !graph.NodeProducesGraphOutput(*node_it) &&
        // Make sure the two nodes do not span execution providers.
        node_it->GetExecutionProviderType() == node.GetExecutionProviderType()) {
          // acceptable if all consumer(s) are gemm node(s) 
          std::vector<const Node*> gemm_nodes = graph_utils::FindChildrenByType(*node_it, "Gemm");
          if (gemm_nodes.size() == node_it->GetOutputEdgesCount()) {
            return true;
          }
    }
  }

  // Fusion can be applied if there is a Transpose at the output of Gemm
  // by the rule (AB)' = B'A' provided that C is missing
  // Supported for Opset >=11 as earlier opsets have C as a required input
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {11, 13}) ||
      graph.NodeProducesGraphOutput(node) ||
      // verify that C is missing
      node.InputDefs().size() > 2) {
    return false;
  }

  const auto next_node_it = node.OutputNodesBegin();
  if (next_node_it != node.OutputNodesEnd() &&
      graph_utils::IsSupportedOptypeVersionAndDomain(*next_node_it, "Transpose", {1, 13}) &&
      next_node_it->GetInputEdgesCount() == 1 &&
      // Make sure the two nodes do not span execution providers.
      next_node_it->GetExecutionProviderType() == node.GetExecutionProviderType()) {
    return true;
  }

  // none of the above transpose conditions were met
  return false;
}

}  // namespace onnxruntime
