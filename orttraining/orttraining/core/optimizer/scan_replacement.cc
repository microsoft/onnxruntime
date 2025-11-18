
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/scan_replacement.h"

namespace onnxruntime {
bool ScanReplacement::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  auto outputs = node.OutputDefs();
  auto last_output = outputs.at(outputs.size() - 1);

  auto name = last_output->Name();
  LOGS(logger, INFO) << name;
  if (name.size() < 8)
    return true;
  LOGS(logger, INFO) << name.substr(name.size() - 8);
  return name.substr(name.size() - 8) != "_carries";
}

Status ScanReplacement::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  // Get necessary variables
  NodeAttributes& attributes = node.GetMutableAttributes();
  // Graph body = Graph(graph, node, *attributes.at("body").mutable_g());
  Graph* body = node.GetMutableGraphAttribute("body");
  ORT_ENFORCE(body != nullptr);
  auto n_carries = body->GetInputs().size() - attributes.at("num_scan_inputs").i();

  // Modify body subgraph. Make all carries per step output.
  auto outputs = body->GetOutputs();
  for (size_t i = 0; i < n_carries; i++) {
    const ONNX_NAMESPACE::TypeProto* type = outputs[i]->TypeAsProto();
    std::string carries_name = outputs[i]->Name() + "_carries";
    std::string identity_name = outputs[i]->Name() + "_identity";
    NodeArg* carries = &body->GetOrCreateNodeArg(carries_name, type);
    body->AddNode(body->GenerateNodeName(identity_name), "Identity", "", {body->GetNodeArg(outputs[i]->Name())}, {carries});
    outputs.push_back(carries);
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
    node_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_inputs[i]->Name() + "_carries"), &type));
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

  // *attributes.at("body").mutable_g() = body->ToGraphProto();

  graph.SetGraphResolveNeeded();

  /*
  Graph::ResolveOptions options;
  std::vector<Graph *> additional_graphs = {body};
  options.additional_graphs = &additional_graphs;
  LOGS(logger, INFO) << "ScanReplace resolving";
  body->SetGraphResolveNeeded();
  graph.SetGraphResolveNeeded();
  ORT_RETURN_IF_ERROR(body->Resolve(options));
  LOGS(logger, INFO) << "ScanReplace resolved";
  //*/

  /*
  Node& scan_training = graph.AddNode(graph.GenerateNodeName(node.Name() + std::string("_training")), "Scan", "Modified Scan operation for training",
                          node_inputs,
                          node_outputs,
                          &attributes,
                          kOnnxDomain);
  scan_training.SetExecutionProviderType(node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, scan_training, node);
  */
  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;

  /*
  LOGS(logger, INFO) << "Scan resolving";
  ORT_RETURN_IF_ERROR(body.Resolve());
  LOGS(logger, INFO) << "Scan resolved";
  *attributes.at("body").mutable_g() = body.ToGraphProto(); // Update body subgraph
  //*/

  return Status::OK();
}
}  // namespace onnxruntime
