
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/scan_replacement.h"

namespace onnxruntime {
bool ScanReplacement::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  return true;
}

Status ScanReplacement::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
	// Get necessary variables
	auto attributes = node.GetAttributes();
	auto body = Graph(graph, node, *attributes.at("body").mutable_g());
	auto n_carries = body.GetInputs().size() - attributes.at("num_scan_input").i();

	std::vector<NodeArg *>& node_inputs = node.MutableInputDefs();
	std::vector<NodeArg *>& node_outputs = node.MutableOutputDefs();

	// Modify body subgraph. Make all carries per step output.
	const std::vector<const NodeArg*>& outputs = body.GetOutputs();
	auto modified_outputs = std::vector(outputs);
	for (auto i = 0; i < n_carries; i++)
		modified_outputs.push_back(outputs[i]);
	body.SetOutputs(modified_outputs);
	*attributes.at("body").mutable_g() = body.ToGraphProto(); // Update body subgraph

	// Modify attributes for newly created carries output.
	if (attributes.find("scan_output_directions") != attributes.end())
	{
		auto attr = attributes.at("scan_output_directions");
		for (auto i = 0; i < n_carries; i++)
			*attr.mutable_ints()->Add() = 0;
	}

	// Create a new Scan operation
	for (auto i = 0; i < n_carries; i++)
	{
		ONNX_NAMESPACE::TypeProto type;
		const ONNX_NAMESPACE::TypeProto *carry_type = (*(node_inputs.begin() + i))->TypeAsProto();
		type.mutable_tensor_type()->set_elem_type(carry_type->tensor_type().elem_type());
		node_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_inputs[i]->Name() + "_carries"), nullptr));
	}

	Node& scan_training = graph.AddNode(graph.GenerateNodeName(node.Name() + std::string("_training")), "Scan", "Modified Scan operation for training",
				node_inputs,
				node_outputs,
				&attributes,
				kOnnxDomain);
	scan_training.SetExecutionProviderType(node.GetExecutionProviderType());
	graph_utils::FinalizeNodeFusion(graph, scan_training, node);
	rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

	return Status::OK();
}
}  // namespace onnxruntime
