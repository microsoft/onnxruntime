// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/sum_accumulate_transformer.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

bool SumAccumulateTransformer::SatisfyCondition(const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8, 13})) {
    return false;
  }

  const auto next_node = node.OutputNodesBegin();

  if (next_node != node.OutputNodesEnd() &&
      node.GetOutputEdgesCount() == 1 &&
      graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "InPlaceAccumulator", {1}, kMSDomain) &&
      next_node->InputDefs().size() == 2 &&
      graph_utils::GetNodeOutputName(node, 0) == graph_utils::GetNodeInputName(*next_node, 1)) {
    return true;
  }
  return false;
}

Status SumAccumulateTransformer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Traverse backward from the bottom of the graph, so that the recompute nodes
  // for lower layers are executed earlier
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);

    if (!SatisfyCondition(node)) {
      continue;
    }

    std::vector<NodeArg*>& sum_inputs = node.MutableInputDefs();
    Node& accu_node = *graph.GetNode(node.OutputNodesBegin()->Index());
    NodeArg* buffer_input = accu_node.MutableInputDefs()[0];

    std::vector<NodeArg*> outputs;
    for (NodeArg* sum_input : sum_inputs) {
      NodeArg& accumulated_output = graph.GetOrCreateNodeArg("Accumulated_" + sum_input->Name(),
                                                             buffer_input->TypeAsProto());
      outputs.push_back(&accumulated_output);

      graph.AddNode(node.Name() + "_InPlaceAccumulator_" + sum_input->Name(),
                    "InPlaceAccumulator",
                    "Accumulator for " + sum_input->Name(),
                    {buffer_input, sum_input},
                    {&accumulated_output},
                    nullptr,
                    kMSDomain);
    }

    graph.AddNode(accu_node.Name() + "_" + "DeduplicateBuffer",
                  "DeduplicateBuffer",
                  "DeduplicateBuffer for " + accu_node.Name(),
                  outputs,
                  accu_node.MutableOutputDefs(),
                  nullptr,
                  kMSDomain);

    LOGS(logger, INFO) << "SumAccumulateTransformer was applied on " << node.Name() << " and " << accu_node.Name();

    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, accu_node);
    graph.RemoveNode(accu_node.Index());

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
