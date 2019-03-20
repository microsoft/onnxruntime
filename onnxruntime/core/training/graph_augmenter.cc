// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/graph_augmenter.h"
using namespace std;

namespace onnxruntime {
namespace training {
using namespace onnxruntime::common;

Status GraphAugmenter::AugmentGraph(Graph& graph, const GraphDefs& graph_element_defs) {
  // Add new initializers to the graph.
  for (const auto& tensor_proto : graph_element_defs.Initializers()) {
    graph.AddInitializedTensor(tensor_proto);
  }

  // Add new nodes to the graph.
  for (const auto& node_def : graph_element_defs.NodeDefs()) {
    vector<NodeArg*> input_args, output_args;

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
                  "",
                  input_args,
                  output_args,
                  &node_def.attributes,
                  node_def.domain);
  }

  // Add new outputs to the graph.
  vector<const NodeArg*> new_output_args = graph.GetOutputs();  // Make a copy of existing output args.
  for (const auto& output_name : graph_element_defs.GraphOutputs()) {
    const auto* output_arg = graph.GetNodeArg(output_name);
    ORT_RETURN_IF_NOT(output_arg != nullptr, "Failed to set graph output ", output_name);
    if (std::find(new_output_args.begin(), new_output_args.end(), output_arg) == new_output_args.end()) {
      new_output_args.emplace_back(output_arg);
    }
  }
  graph.SetOutputOrder(new_output_args);  // By setting this, Graph::SetGraphInputsOutputs could infer the output as expected.

  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  return graph.Resolve();
}

}  // namespace training
}  // namespace onnxruntime
