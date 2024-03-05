// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>
#include "orttraining/core/optimizer/compute_optimizer/upstream_pad.h"

#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/compute_optimizer/upstream_pad.h"

namespace onnxruntime {

Status UpstreamPad::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                              const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (!p_node || p_node->OpType() != "PythonOp") {
      continue;
    }
    // if output nodes count less than 2, then it is not a valid PadAndUnflatten node
    if (p_node->GetOutputEdgesCount() < 2) {
      continue;
    }

    bool output_all_padding = false;

    for (auto iter = p_node->OutputNodesBegin(); iter != p_node->OutputNodesEnd(); ++iter) {
      Node& output_node = *graph.GetNode(iter->Index());
      if (output_node.OpType() == "PadAndUnflatten") {
        output_all_padding = true;
      } else {
        output_all_padding = false;
        break;
      }
    }

    if (!output_all_padding) {
      continue;
    }

    Node* first_output_node = graph.GetNode(p_node->OutputNodesBegin()->Index());
    ORT_ENFORCE(first_output_node->OpType() == "PadAndUnflatten");

    InlinedVector<NodeArg*> pad_node_input_args;
    pad_node_input_args.reserve(3);
    pad_node_input_args.push_back(p_node->MutableInputDefs()[0]);
    pad_node_input_args.push_back(first_output_node->MutableInputDefs()[1]);
    pad_node_input_args.push_back(first_output_node->MutableInputDefs()[2]);

    InlinedVector<NodeArg*> pad_node_output_args;
    pad_node_output_args.push_back(
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padded_result"),
                                  &(*first_output_node->MutableOutputDefs()[0]->TypeAsProto())));
    // pad_node_output_args.push_back(
    //     &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padded_d1xd2_shape"),
    //                               nullptr));

    Node* new_gathergrad_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
        graph, *p_node,
        0,
        0 /* new_node_input_index*/,
        0 /* new_node_output_index*/,
        graph.GenerateNodeName("upPaddingRecover"),
        "PadAndUnflatten",
        "PadAndUnflatten node to recover invalid tokens.",
        pad_node_input_args,
        pad_node_output_args,
        {},
        kMSDomain,
        logger);

    new_gathergrad_node->SetExecutionProviderType(p_node->GetExecutionProviderType());

    if(first_output_node->MutableOutputDefs()[0]->Shape()){
      new_gathergrad_node->MutableOutputDefs()[0]->SetShape(*first_output_node->MutableOutputDefs()[0]->Shape());
      p_node->MutableOutputDefs()[1]->SetShape(*first_output_node->MutableOutputDefs()[0]->Shape());
    }

    InlinedVector<Node*> to_remove_nodes;
    for (auto iter = p_node->OutputNodesBegin(); iter != p_node->OutputNodesEnd(); ++iter) {
      if (graph.GetNode(iter->Index())) {
        to_remove_nodes.push_back(graph.GetNode(iter->Index()));
      }
    }
    for (auto output_node : to_remove_nodes) {
      graph_utils::ReplaceDownstreamNodeInput(graph, *output_node, 0 /*output_idx*/, *p_node,
                                              1 /*replacement_output_idx*/);

      auto origin_consumer_nodes = graph.GetConsumerNodes(output_node->MutableOutputDefs()[0]->Name());
      std::vector<Node*> op_consumers;
      op_consumers.reserve(origin_consumer_nodes.size());
      for (auto& consumer_node : origin_consumer_nodes) {
        op_consumers.push_back(graph.GetNode(consumer_node->Index()));
      }
      graph.UpdateConsumerNodes(p_node->OutputDefs()[1]->Name(), op_consumers);
      graph.RemoveNode(output_node->Index());
    }
    // p_node->MutableInputDefs()[0]->ClearShape();
    p_node->MutableOutputDefs()[1]->ClearShape();

    auto& attributes = p_node->GetMutableAttributes();
    ORT_ENFORCE(attributes.find("output_tensor_ranks") != attributes.end());
    auto& origin_output_tensor_ranks = attributes.at("output_tensor_ranks").ints();
    std::vector<int64_t> output_tensor_ranks{origin_output_tensor_ranks.cbegin(), origin_output_tensor_ranks.cend()};
    ORT_ENFORCE(output_tensor_ranks.size() == 1 && output_tensor_ranks[0] >= 2);
    output_tensor_ranks[0] = 3;
    attributes["output_tensor_ranks"] = ONNX_NAMESPACE::MakeAttribute("output_tensor_ranks", output_tensor_ranks);
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
