// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {
using namespace onnxruntime::common;

Status GraphAugmenter::AugmentGraph(Graph& graph, const GraphDefs& graph_element_defs) {
  // Add new initializers to the graph. - no op if it already exists
  for (const auto& tensor_proto : graph_element_defs.Initializers()) {
    const ONNX_NAMESPACE::TensorProto* exist_initializer = nullptr;
    if (!graph.GetInitializedTensor(tensor_proto.name(), exist_initializer)) {
      graph.AddInitializedTensor(tensor_proto);
      graph.GetOrCreateNodeArg(tensor_proto.name(), nullptr);
    }
  }

  // Add new nodes to the graph.
  for (const auto& node_def : graph_element_defs.NodeDefs()) {
    std::vector<NodeArg*> input_args, output_args;

    for (const auto& arg : node_def.input_args) {
      NodeArg& node_arg = graph.GetOrCreateNodeArg(arg.name, arg.type_proto);
      input_args.push_back(&node_arg);
    }

    for (const auto& arg : node_def.output_args) {
      NodeArg& node_arg = graph.GetOrCreateNodeArg(arg.name, arg.type_proto);
      output_args.push_back(&node_arg);
    }

    graph.AddNode(node_def.name,
                  node_def.op_type,
                  "Backward pass",
                  input_args,
                  output_args,
                  &node_def.attributes,
                  node_def.domain);
  }

  // Add new outputs to the graph.
  std::vector<const NodeArg*> new_output_args = graph.GetOutputs();  // Make a copy of existing output args.
  for (const auto& output_name : graph_element_defs.GraphOutputs()) {
    const auto* output_arg = graph.GetNodeArg(output_name);

    ORT_RETURN_IF(output_arg == nullptr, "Failed to set graph output ", output_name);
    if (std::find(new_output_args.begin(), new_output_args.end(), output_arg) == new_output_args.end()) {
      new_output_args.emplace_back(output_arg);
    }
  }
  graph.SetOutputs(new_output_args);  // By setting this, Graph::SetGraphInputsOutputs could infer the output as expected.

  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  return graph.Resolve();
}

Status GraphAugmenter::OverrideGraphOutputs(Graph& graph, const std::vector<std::string>& graph_outputs) {
  {
    std::unordered_set<std::string> unique_graph_outputs(graph_outputs.begin(), graph_outputs.end());
    if (unique_graph_outputs.size() != graph_outputs.size()) {
      std::string error_message{"The specified graph outputs are not unique:"};
      for (const auto& graph_output : graph_outputs) {
        error_message += "\n  " + graph_output;
      }
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, error_message);
    }
  }

  std::vector<const NodeArg*> new_output_args;
  for (const auto& output_name : graph_outputs) {
    const auto* output_arg = graph.GetNodeArg(output_name);
    ORT_RETURN_IF(output_arg == nullptr, "Failed to set graph output ", output_name);
    new_output_args.emplace_back(output_arg);
  }

  graph.SetOutputs(new_output_args);  // By setting this, Graph::SetGraphInputsOutputs could infer the output as expected.
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  return graph.Resolve();
}

}  // namespace training
}  // namespace onnxruntime
