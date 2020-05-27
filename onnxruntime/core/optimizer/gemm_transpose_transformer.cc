// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gemm_transpose_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status GemmTransposeTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    auto& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {7, 9}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& node_consumers = graph.GetConsumerNodes(node.MutableOutputDefs()[0]->Name());
    if (node_consumers.empty() && node_consumers.size() != 1) {
      continue;
    }

    auto trans_node = const_cast<Node*>(node_consumers[0]);
    // if the node has Graph output, skip it too
    if (trans_node->OpType() != "Transpose" || !graph.GetNodeOutputsInGraphOutputs(*trans_node).empty()) {
      continue;
    }

    auto trans_a = node.GetAttributes().at("transA").i();
    auto trans_b = node.GetAttributes().at("transB").i();

    if (trans_a != 1 || trans_b != 0) {
      continue;
    }

    auto perms = RetrieveValues<int64_t>(trans_node->GetAttributes().at("perm"));
    int64_t rank = perms.size();
    if (rank != 2 || perms[0] != 1 || perms[1] != 0) {
      continue;
    }

    NodeArg* left_input = node.MutableInputDefs()[0];
    NodeArg* right_input = node.MutableInputDefs()[1];
    //const std::vector<NodeArg*> input_defs{right_input, left_input, node.MutableInputDefs()[2]};

    std::cout << "find one " << node.Name() << "," << right_input->Name() << ":" << left_input->Name() << std::endl;

    const std::vector<NodeArg*> new_input_defs{right_input, left_input, node.MutableInputDefs()[2]};
    auto gemm_output_type_proto = *trans_node->MutableOutputDefs()[0]->TypeAsProto();
    auto& gemm_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("gemm_fused_2020"), &gemm_output_type_proto);
    Node& gemm_node = graph.AddNode(graph.GenerateNodeName("gemm_fused_2020"),
                                    "Gemm",
                                    "fused Gemm and Transpose ",
                                    new_input_defs,
                                    {&gemm_out_arg});

    gemm_node.AddAttribute("transA", static_cast<int64_t>(1));
    gemm_node.AddAttribute("transB", static_cast<int64_t>(0));
    gemm_node.AddAttribute("beta", static_cast<float>(0.0f));
    // Assign provider to this new node. Provider should be same as the provider for old node.
    gemm_node.SetExecutionProviderType(node.GetExecutionProviderType());
    //Node& trans_node_2 = *trans_node;
    //trans_node->MutableOutputDefs()[0]->ClearShape();
    //graph_utils::FinalizeNodeFusion(graph, {node, trans_node_2}, gemm_node);
    graph_utils::ReplaceDownstreamNodeInput(graph, *trans_node, 0, gemm_node, 0);
    removed_nodes.push_front(node.Index());
    removed_nodes.push_front(trans_node->Index());
    modified = true;
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
    graph.SetGraphResolveNeeded();
    auto ret = graph.Resolve();
    ORT_ENFORCE(ret.IsOK());
  }

  return Status::OK();
}
}  // namespace onnxruntime
