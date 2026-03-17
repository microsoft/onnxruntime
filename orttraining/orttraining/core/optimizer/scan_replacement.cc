
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/scan_replacement.h"

namespace onnxruntime {
bool ScanReplacement::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  auto body = node.GetGraphAttribute("body");
  if (body->Name().find("ScanReplacement/") == 0)
    return false;
  return true;
}

// ScanReplacement modifies outputs of Scan operation.
// Let say inputs are carries-in and scan-inputs, and outputs are carries-out and scan-outputs.
// ScanReplacement modifis outputs as carries-out, carries-out, scan-outputs, and intermediate-value.
// Where second carries-out is for carries per step, and intermediate-value is to avoid gradient checkpointing like behavior in the ScanGrad.
// To distinguish scan-outputs and intermediate-value, scan_output_axes for intermediate-value is chosen different value from scan-outputs[-1].
Status ScanReplacement::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  // Get necessary variables
  NodeAttributes& attributes = node.GetMutableAttributes();
  Graph* body = node.GetMutableGraphAttribute("body");
  ORT_ENFORCE(body != nullptr);
  auto n_carries = body->GetInputs().size() - attributes.at("num_scan_inputs").i();
  auto& body_inputs = body->GetInputs();

  // Modify body subgraph. Make all carries per step output.
  auto& inputs = body->GetInputs();
  auto outputs = body->GetOutputs();
  std::vector<NodeArg*> carries;
  carries.reserve(n_carries);
  for (size_t i = 0; i < n_carries; i++) {
    const ONNX_NAMESPACE::TypeProto* type = inputs[i]->TypeAsProto();
    std::string input_carries_name = body->GenerateNodeArgName(inputs[i]->Name() + "_carries");
    std::string input_identity_name = body->GenerateNodeArgName(inputs[i]->Name() + "_identity");
    NodeArg* input_carries = &body->GetOrCreateNodeArg(input_carries_name, type);
    body->AddNode(body->GenerateNodeName(input_identity_name), "Identity", "", {body->GetNodeArg(inputs[i]->Name())}, {input_carries});
    carries.push_back(input_carries);
  }
  outputs.insert(outputs.begin() + n_carries, carries.begin(), carries.end());

  std::unordered_set<const NodeArg*> activation;
  std::unordered_set<const Node*> pending;
  for (auto output : outputs) {
    const Node* node = body->GetProducerNode(output->Name());
    if (node == nullptr)
      continue;

    for (auto input : node->InputDefs())
      pending.insert(body->GetProducerNode(input->Name()));

    while (pending.size() > 0) {
      std::unordered_set<const NodeArg*> current_activation;
      for (auto pnode : pending) {
        if (pnode == nullptr)
          continue;

        for (auto i : pnode->InputDefs()) {
          if (body->GetProducerNode(i->Name()))
            current_activation.insert(i);
        }
      }

      pending.clear();
      for (auto a : current_activation)
        pending.insert(body->GetProducerNode(a->Name()));
      activation.insert(current_activation.begin(), current_activation.end());
    }
  }

  for (auto a : activation)
    outputs.push_back(a);

  body->SetOutputs(outputs);
  body->SetGraphProtoSyncNeeded();
  body->SetGraphResolveNeeded();
  body->SetName("ScanReplacement/" + body->Name());

  // Create a new Scan operation
  auto& node_inputs = node.MutableInputDefs();
  auto& node_outputs = node.MutableOutputDefs();
  std::vector<NodeArg*> node_carries;
  node_carries.reserve(n_carries);
  for (size_t i = 0; i < n_carries; i++) {
    const ONNX_NAMESPACE::TypeProto* carry_type = node_outputs[i]->TypeAsProto();
    ORT_ENFORCE(carry_type != nullptr);
    ONNX_NAMESPACE::TypeProto type = *carry_type;
    type.mutable_tensor_type()->clear_shape();
    node_carries.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_outputs[i]->Name() + "_carries"), &type));
  }
  node_outputs.insert(node_outputs.begin() + n_carries, node_carries.begin(), node_carries.end());

  for (auto& a : activation) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(a->TypeAsProto()->tensor_type().elem_type());
    node_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(a->Name()), &type_proto));
  }

  // Modify attributes
  if (attributes.find("scan_output_axes") != attributes.end()) {
    auto& attr = attributes.at("scan_output_axes");
    std::vector<int64_t> scan_output_axes(attr.mutable_ints()->begin(), attr.mutable_ints()->end());
    for (size_t i = 0; i < n_carries; i++)
      scan_output_axes.insert(scan_output_axes.begin() + n_carries, 0);
    if (scan_output_axes.back() == 0)
      for (size_t i = 0; i < activation.size(); i++)
        scan_output_axes.push_back(-1);
    else
      for (size_t i = 0; i < activation.size(); i++)
        scan_output_axes.push_back(0);
    attr.mutable_ints()->Assign(scan_output_axes.begin(), scan_output_axes.end());
  } else {
    std::vector<int64_t> scan_output_axes;
    size_t osize = body->GetOutputs().size() - n_carries;
    scan_output_axes.reserve(osize);
    for (size_t i = 0; i < osize; i++)
      if (i < (osize - activation.size()))
        scan_output_axes.push_back(0);
      else
        scan_output_axes.push_back(-1);
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name("scan_output_axes");
    attr.mutable_ints()->Assign(scan_output_axes.begin(), scan_output_axes.end());
    attributes.insert(std::make_pair("scan_output_axes", attr));
  }
  if (attributes.find("scan_output_directions") != attributes.end()) {
    auto& attr = attributes.at("scan_output_directions");
    std::vector<int64_t> scan_output_directions(attr.mutable_ints()->begin(), attr.mutable_ints()->end());
    for (size_t i = 0; i < n_carries; i++)
      scan_output_directions.insert(scan_output_directions.begin() + n_carries, 0);
    for (size_t i = 0; i < activation.size(); i++)
      scan_output_directions.push_back(0);
    attr.mutable_ints()->Assign(scan_output_directions.begin(), scan_output_directions.end());
  }

  graph.SetGraphResolveNeeded();
  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;

  return Status::OK();
}
}  // namespace onnxruntime
