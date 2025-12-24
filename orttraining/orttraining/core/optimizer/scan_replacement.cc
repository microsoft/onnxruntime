
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/scan_replacement.h"

namespace onnxruntime {
bool ScanReplacement::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  auto outputs = node.OutputDefs();
  auto last_output = outputs.at(outputs.size() - 1);

  auto name = last_output->Name();
  if (name.size() < 8)
    return true;
  return name.substr(name.size() - 8) != "_carries";
}

Status ScanReplacement::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  // Get necessary variables
  NodeAttributes& attributes = node.GetMutableAttributes();
  Graph* body = node.GetMutableGraphAttribute("body");
  ORT_ENFORCE(body != nullptr);
  auto n_carries = body->GetInputs().size() - attributes.at("num_scan_inputs").i();

  // Modify body subgraph. Make all carries per step output.
  auto& inputs = body->GetInputs();
  auto outputs = body->GetOutputs();
  for (size_t i = 0; i < n_carries; i++) {
    const ONNX_NAMESPACE::TypeProto* type = inputs[i]->TypeAsProto();
    std::string input_carries_name = inputs[i]->Name() + "_carries";
    std::string input_identity_name = inputs[i]->Name() + "_identity";
    NodeArg* input_carries = &body->GetOrCreateNodeArg(input_carries_name, type);
    body->AddNode(body->GenerateNodeName(input_identity_name), "Identity", "", {body->GetNodeArg(inputs[i]->Name())}, {input_carries});
    outputs.push_back(input_carries);
  }

  body->SetOutputs(outputs);
  body->SetGraphProtoSyncNeeded();
  body->SetGraphResolveNeeded();

  // Create a new Scan operation
  auto& node_inputs = node.MutableInputDefs();
  auto& node_outputs = node.MutableOutputDefs();
  for (size_t i = 0; i < n_carries; i++) {
    const ONNX_NAMESPACE::TypeProto* carry_type = node_outputs[i]->TypeAsProto();
    ORT_ENFORCE(carry_type != nullptr);
    ONNX_NAMESPACE::TypeProto type = *carry_type;
    type.mutable_tensor_type()->clear_shape();
    node_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_outputs[i]->Name() + "_carries"), &type));
  }

  // Modify attributes
  if (attributes.find("scan_output_axes") != attributes.end()) {
    auto& attr = attributes.at("scan_output_axes");
    for (size_t i = 0; i < n_carries; i++)
      attr.add_ints(0);
  }
  if (attributes.find("scan_output_directions") != attributes.end()) {
    auto& attr = attributes.at("scan_output_directions");
    for (size_t i = 0; i < n_carries; i++)
      attr.add_ints(0);
  }

  graph.SetGraphResolveNeeded();
  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;

  return Status::OK();
}
}  // namespace onnxruntime
