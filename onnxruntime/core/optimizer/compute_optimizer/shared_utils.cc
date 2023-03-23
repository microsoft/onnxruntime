// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE

#include <onnx/defs/attr_proto_util.h>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

Node* InsertIntermediateNodeOnDestInput(Graph& graph,
                                        Node& dest_node, int dest_in_index,
                                        int new_node_input_index,
                                        int new_node_output_index,
                                        const std::string& name, const std::string& op_type,
                                        const std::string& description,
                                        const InlinedVector<NodeArg*>& input_args,
                                        const InlinedVector<NodeArg*>& output_args,
                                        const onnxruntime::NodeAttributes& attributes,
                                        const std::string& domain,
                                        const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "Inserting " + op_type + " node on " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input " +
                             dest_node.InputDefs()[dest_in_index]->Name() + ", and connect inserted node's " +
                             std::to_string(new_node_output_index) + "th output to " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input.");

  ORT_ENFORCE(dest_in_index < static_cast<int>(dest_node.InputDefs().size()));
  ORT_ENFORCE(new_node_input_index < static_cast<int>(input_args.size()), "new_node_input_index is out of range.");
  ORT_ENFORCE(new_node_output_index < static_cast<int>(output_args.size()), "new_node_output_index is out of range.");
  ORT_ENFORCE(dest_node.MutableInputDefs()[dest_in_index] == input_args[new_node_input_index],
              "input_args[new_node_input_index] is not the same as dest_node.MutableInputDefs()[dest_in_index].",
              dest_node.MutableInputDefs()[dest_in_index]->Name(), " vs ", input_args[new_node_input_index]->Name());

  // Prepare Input and Outputs for the duplicated Gather/GatherND node.
  NodeArg* src_node_arg = dest_node.MutableInputDefs()[dest_in_index];

  // Create the duplicated Gather/GatherND node.
  Node& new_node = graph.AddNode(name, op_type, description, input_args, output_args, &attributes, domain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(new_node), "Failed to set op schema for " + new_node.Name());

  // Connect dest_node's input node to duplicated node.
  // Update new node producer and consumer map.
  for (size_t j = 0; j < new_node.MutableOutputDefs().size(); ++j) {
    graph.UpdateProducerNode(new_node.MutableOutputDefs()[j]->Name(), new_node.Index());
  }
  graph.AddConsumerNode(src_node_arg->Name(), &new_node);
  const Node* src_node = graph.GetProducerNode(src_node_arg->Name());
  if (src_node) {
    int src_out_index = optimizer_utils::IndexOfNodeOutput(*src_node, *src_node_arg);
    graph.AddEdge(src_node->Index(), new_node.Index(), src_out_index, new_node_input_index);
  }

  // Remove edge between dest_node and src_node.
  // Be noted, this will remove dest_node's input edges to src_node
  // (and also the src_node's output edges to dest_node).
  std::vector<graph_utils::GraphEdge> input_edge_to_remove;
  input_edge_to_remove.reserve(1);
  for (auto it = dest_node.InputEdgesBegin(), end = dest_node.InputEdgesEnd(); it != end; ++it) {
    LOG_DEBUG_INFO(logger, "dest_node " + dest_node.Name() + " input edge: " + it->GetNode().Name() +
                               " output index: " + std::to_string(it->GetSrcArgIndex()) + " input index: " +
                               std::to_string(it->GetDstArgIndex()));
    if (it->GetDstArgIndex() == dest_in_index) {
      input_edge_to_remove.push_back(graph_utils::GraphEdge::CreateGraphEdge(dest_node, *it, true));
      break;
    }
  }

  // If the input is graph input or initializer, no edge will be removed.
  if (input_edge_to_remove.size() > 0) {
    graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edge_to_remove);

    // Remove target node from target input arg's consumer list.
    const std::string& src_node_arg_name = src_node_arg->Name();
    int input_use_count_by_dest_node = 0;
    for (size_t i = 0; i < dest_node.InputDefs().size(); ++i) {
      if (dest_node.InputDefs()[i]->Name().compare(src_node_arg_name) == 0) {
        ++input_use_count_by_dest_node;
      }
    }

    if (input_use_count_by_dest_node == 1) {
      graph.RemoveConsumerNode(src_node_arg_name, &dest_node);
    }
  }

  // Connect duplicated gather node to target node's input.
  dest_node.MutableInputDefs()[dest_in_index] = new_node.MutableOutputDefs()[new_node_output_index];
  // Add new edge connecting the duplicated gather with the target node directly.
  // This also updates the destination node's input node args
  graph.AddEdge(new_node.Index(), dest_node.Index(), new_node_output_index, dest_in_index);
  graph.AddConsumerNode(new_node.MutableOutputDefs()[new_node_output_index]->Name(), &dest_node);
  LOG_DEBUG_INFO(logger, "Inserted " + op_type + " node on " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input " +
                             dest_node.InputDefs()[dest_in_index]->Name());
  return &new_node;
}

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
