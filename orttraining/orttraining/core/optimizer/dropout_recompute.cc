// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/dropout_recompute.h"
#include "orttraining/core/graph/recompute_graph_utils.h"

namespace onnxruntime {

Node& InsertDropoutRecompute(Graph& graph, Node& node, bool use_original_input) {
  NodeArg* input = node.MutableInputDefs()[0];
  if (!use_original_input) {
    auto& recomputed_input = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(input->Name()),
                                                      input->TypeAsProto());
    input = &recomputed_input;
  }

  const auto& output = node.OutputDefs()[0];
  auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                     output->TypeAsProto());

  Node& recompute_node = graph.AddNode(node.Name() + "_recompute",
                                       "DropoutGrad",
                                       "Recompute of " + node.Name(),
                                       {
                                           input,                        // X
                                           node.MutableOutputDefs()[1],  // mask
                                           node.MutableInputDefs()[1],   // ratio
                                           node.MutableInputDefs()[2]    // training_mode

                                       },
                                       {&recomputed_output},
                                       {},
                                       kMSDomain);

  return recompute_node;
}

}  // namespace onnxruntime
