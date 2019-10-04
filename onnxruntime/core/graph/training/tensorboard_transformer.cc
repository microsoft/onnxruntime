// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/training/tensorboard_transformer.h"
#include "core/graph/training/attr_proto_util.h"
#include "core/graph/training/graph_augmenter.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace training {

Status TransformGraphForTensorboard(Graph& graph,
                                    const std::string& summary_name,
                                    const std::vector<std::string>& scalar_nodes,
                                    const std::vector<std::string>& histogram_nodes) {
  std::vector<ArgDef> summary_args;
  std::vector<NodeDef> new_nodes;

  // SummaryScalar nodes.
  for (const std::string& scalar_input : scalar_nodes) {
    std::string scalar_output = summary_name + "/scalar/" + scalar_input;
    summary_args.push_back(ArgDef(scalar_output));
    new_nodes.emplace_back(NodeDef("SummaryScalar",
                                   {ArgDef(scalar_input)},
                                   {ArgDef(scalar_output)},
                                   {MakeAttribute("tags", std::vector<std::string>{scalar_output})},
                                   scalar_output));
  }

  // SummaryHistogram nodes.
  for (const std::string& histogram_input : histogram_nodes) {
    std::string histogram_output = summary_name + "/histogram/" + histogram_input;
    summary_args.push_back(ArgDef(histogram_output));
    new_nodes.emplace_back(NodeDef("SummaryHistogram",
                                   {ArgDef(histogram_input)},
                                   {ArgDef(histogram_output)},
                                   {MakeAttribute("tag", histogram_output)},
                                   histogram_output));
  }

  // SummaryMerge (if any tensorboard nodes exist).
  if (summary_args.size() > 0) {
    new_nodes.emplace_back(NodeDef("SummaryMerge",
                                    summary_args,             // Inputs
                                    {ArgDef(summary_name)}, // Outputs
                                    NodeAttributes(),
                                    summary_name));

    // Modify graph.
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(new_nodes);
    graph_defs.AddGraphOutputs({summary_name});
    return GraphAugmenter::AugmentGraph(graph, graph_defs);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
