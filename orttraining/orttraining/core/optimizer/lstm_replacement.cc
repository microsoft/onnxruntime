// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/lstm_replacement.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status LSTMReplacement::Apply(Graph& graph, Node& lstm_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const auto& lstm_inputs = lstm_node.MutableInputDefs();
  auto& lstm_outputs = lstm_node.MutableOutputDefs();

  ONNX_NAMESPACE::TypeProto lstm_input_type;
  lstm_input_type.mutable_tensor_type()->set_elem_type(lstm_inputs.front()->TypeAsProto()->tensor_type().elem_type());

  if (lstm_outputs.empty() || !lstm_outputs[0]->Exists()) {
    // Add all LSTM cell optional outputs for computation even if not required to ensure all outputs are computed
    // so that gradient computation can pick them up.
    NodeArg& all_hidden_states = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("all_hidden_states"),
                                                          &lstm_input_type);
    if (!lstm_outputs.empty()) {
      lstm_outputs[0] = &all_hidden_states;
    } else {
      lstm_outputs.push_back(&all_hidden_states);
    }
  }

  if (lstm_outputs.size() == 1U) {
    NodeArg& placeholder = graph.GetOrCreateNodeArg("", nullptr);
    lstm_outputs.push_back(&placeholder);
  }

  if (lstm_outputs.size() == 2U) {
    NodeArg& placeholder = graph.GetOrCreateNodeArg("", nullptr);
    lstm_outputs.push_back(&placeholder);
  }

  NodeArg& all_cell_states = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("all_cell_states"),
                                                      &lstm_input_type);
  lstm_outputs.push_back(&all_cell_states);
  NodeArg& iofc = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("iofc"), &lstm_input_type);
  lstm_outputs.push_back(&iofc);

  // LSTMTraining should have 5 outputs
  ORT_ENFORCE(lstm_node.OutputDefs().size() == 5U);

  auto lstm_attributes = lstm_node.GetAttributes();
  lstm_attributes.erase("layout");

  Node& lstm_training_node = graph.AddNode(graph.GenerateNodeName(lstm_node.Name() + "_training"),
                                           "LSTMTraining",
                                           "LSTM with extra outputs for needed for gradient computation.",
                                           lstm_inputs,
                                           lstm_outputs,
                                           &lstm_attributes,  // AddNode makes a copy, so ok to pass pointer to local var
                                           kMSDomain);

  // Assign provider to this new node. Provider should be same as the provider for old node.
  lstm_training_node.SetExecutionProviderType(lstm_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, lstm_training_node, lstm_node);
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

bool LSTMReplacement::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  return true;
}

}  // namespace onnxruntime
