// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constant_folding.h"
#include "core/graph/graph_utils.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

Status ConstantFolding::Apply(Graph& graph, Node& node, bool& modified) {
  // TODO Can we reuse the model across calls of the constant folding rule?
  // TODO Instead of computing one node at a time, we can traverse the whole graph and
  // find constant parts of it. Then we can create bigger subgraphs to compute directly.
  // This will most likely be done with a Transformer rather than a RewriteRule.
  auto p_model = std::make_unique<onnxruntime::Model>("SubgraphToCompute", false, ModelMetaData(),
                                                      IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}),
                                                      graph.DomainToVersionMap());
  Graph& subgraph = p_model->MainGraph();

  std::vector<onnxruntime::NodeIndex> subgraph_nodes;
  subgraph_nodes.push_back(node.Index());

  // Build the subgraph.
  graph_edit_utils::BuildSubgraph(graph, subgraph_nodes, subgraph);

  SessionOptions so;
  so.session_logid = "SubgraphComputation";
  InferenceSession session_object{so};
  // TODO Can we directly pass the model to the session object instead of having to dump it to a stream?
  std::stringstream model_istream;
  p_model->ToProto().SerializeToOstream(&model_istream);
  ONNXRUNTIME_RETURN_IF_ERROR(session_object.Load(model_istream));
  ONNXRUNTIME_RETURN_IF_ERROR(session_object.Initialize());

  // Prepare outputs. TODO Check how this will work with multiple outputs.
  std::vector<std::string> output_names;
  for (auto& output : subgraph.GetOutputs()) {
    output_names.push_back(output->Name());
  }
  // No inputs needed as they are all initializers.
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<MLValue> fetches;

  // Now run.
  Status st = session_object.Run(feeds, output_names, &fetches);

  MLValue& mlvalue = fetches[0];
  

  return Status::OK();
}  // namespace onnxruntime

bool ConstantFolding::SatisfyCondition(const Graph& graph, const Node& node) {
  return graph_edit_utils::IsConstantInputsNode(graph, node);
}

}  // namespace onnxruntime
