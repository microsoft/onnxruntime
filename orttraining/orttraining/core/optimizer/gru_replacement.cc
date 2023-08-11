// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/gru_replacement.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status GRUReplacement::Apply(Graph& graph, Node& gru_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const auto& gru_inputs = gru_node.MutableInputDefs();
  auto& gru_outputs = gru_node.MutableOutputDefs();

  const auto* type_proto = gru_inputs.front()->TypeAsProto();
  ORT_ENFORCE(type_proto && type_proto->has_tensor_type() && type_proto->tensor_type().has_elem_type(),
              "Could not decipher the type of input to GRU node: ", gru_node.Name());

  ONNX_NAMESPACE::TypeProto gru_input_type;
  gru_input_type.mutable_tensor_type()->set_elem_type(type_proto->tensor_type().elem_type());

  if (gru_outputs.empty() || !gru_outputs[0]->Exists()) {
    // Add all GRU cell optional outputs for computation even if not required to ensure all outputs are computed
    // so that gradient computation can pick them up.
    NodeArg& all_hidden_states = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("all_hidden_states"),
                                                          &gru_input_type);
    if (!gru_outputs.empty()) {
      gru_outputs[0] = &all_hidden_states;
    } else {
      gru_outputs.push_back(&all_hidden_states);
    }
  }

  if (gru_outputs.size() == 1U) {
    NodeArg& placeholder = graph.GetOrCreateNodeArg("", nullptr);
    gru_outputs.push_back(&placeholder);
  }

  NodeArg& zrh = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("zrh"), &gru_input_type);
  gru_outputs.push_back(&zrh);

  // GRUTraining should have 3 outputs
  ORT_ENFORCE(gru_node.OutputDefs().size() == 3U);

  auto gru_attributes = gru_node.GetAttributes();
  gru_attributes.erase("layout");

  Node& gru_training_node = graph.AddNode(graph.GenerateNodeName(gru_node.Name() + "_training"),
                                          "GRUTraining",
                                          "GRU with extra outputs needed for gradient computation.",
                                          gru_inputs,
                                          gru_outputs,
                                          &gru_attributes,  // AddNode makes a copy, so ok to pass pointer to local var
                                          kMSDomain);

  // Assign provider to this new node. Provider should be same as the provider for the node being replaced.
  gru_training_node.SetExecutionProviderType(gru_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, gru_training_node, gru_node);
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

bool GRUReplacement::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  return true;
}

}  // namespace onnxruntime
