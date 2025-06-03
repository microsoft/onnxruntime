// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ep_api_types.h"

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"

namespace onnxruntime {
const std::string& EpNode::Name() const { return node.Name(); }
const std::string& EpNode::OpType() const { return node.OpType(); }
const std::string& EpNode::Domain() const { return node.Domain(); }

Status EpNode::GetInputs(InlinedVector<const OrtValueInfo*>& result) const {
  result.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    result[i] = inputs[i];
  }
  return Status::OK();
}

Status EpNode::GetOutputs(InlinedVector<const OrtValueInfo*>& result) const {
  result.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    result[i] = outputs[i];
  }
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
static Status GetInputOrOutputIndices(gsl::span<const EpValueInfo* const> value_infos,
                                      const std::string& value_info_name,
                                      /*out*/ std::vector<size_t>& indices) {
  indices.reserve(value_infos.size());

  bool found = false;
  for (size_t i = 0; i < value_infos.size(); i++) {
    if (value_infos[i]->Name() == value_info_name) {
      indices.push_back(i);
      found = true;
    }
  }

  ORT_RETURN_IF_NOT(found, "Did not find OrtValueInfo with name ", value_info_name);
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

Status EpValueInfo::GetProducerInfo(OrtValueInfo::ProducerInfo& producer_info) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  producer_info.node = nullptr;
  producer_info.output_index = 0;

  const Node* node = graph.graph_viewer.GetProducerNode(name);
  if (node == nullptr) {
    return Status::OK();
  }

  auto it = graph.index_to_node.find(node->Index());
  if (it == graph.index_to_node.end()) {
    return Status::OK();  // Node is not in this GraphViewer
  }

  const EpNode* ep_node = it->second;
  std::vector<size_t> output_indices;
  ORT_RETURN_IF_ERROR(GetInputOrOutputIndices(ep_node->outputs, name, output_indices));
  assert(output_indices.size() == 1);  // An output can only come from one node.

  producer_info.node = ep_node->ToExternal();
  producer_info.output_index = output_indices[0];
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(producer_info);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting producers from OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetUses(std::vector<OrtValueInfo::UseInfo>& uses) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  std::vector<const Node*> nodes = graph.graph_viewer.GetConsumerNodes(name);
  if (nodes.empty()) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(uses.empty(), "Internal error: uses should be empty in GetUses()");
  uses.reserve(nodes.size());
  for (const Node* node : nodes) {
    auto it = graph.index_to_node.find(node->Index());
    if (it == graph.index_to_node.end()) {
      continue;  // Node is not in this GraphViewer
    }

    const EpNode* ep_node = it->second;
    std::vector<size_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputOrOutputIndices(ep_node->inputs, name, input_indices));

    for (size_t input_index : input_indices) {
      OrtValueInfo::UseInfo use_info(ep_node->ToExternal(), input_index);
      uses.push_back(use_info);
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(uses);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting uses of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetNumUses(size_t& num_uses) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  num_uses = 0;

  std::vector<const Node*> nodes = graph.graph_viewer.GetConsumerNodes(name);
  if (nodes.empty()) {
    return Status::OK();
  }

  for (const Node* node : nodes) {
    auto it = graph.index_to_node.find(node->Index());
    if (it == graph.index_to_node.end()) {
      continue;  // Node is not in this GraphViewer
    }

    const EpNode* ep_node = it->second;
    std::vector<size_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputOrOutputIndices(ep_node->inputs, name, input_indices));

    num_uses += input_indices.size();  // A single OrtNode can use an OrtValueInfo as an input more than once.
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(num_uses);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting uses of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

static EpValueInfo* AddValueInfo(std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos,
                                 const NodeArg& node_arg, const EpGraph& ep_graph) {
  auto it = value_infos.find(node_arg.Name());
  if (it != value_infos.end()) {
    return it->second.get();
  }

  const auto* type_proto = node_arg.TypeAsProto();
  std::unique_ptr<OrtTypeInfo> type_info = type_proto != nullptr ? OrtTypeInfo::FromTypeProto(*type_proto)
                                                                 : nullptr;
  auto ep_value_info = std::make_unique<EpValueInfo>(ep_graph, node_arg.Name(), std::move(type_info));
  EpValueInfo* result = ep_value_info.get();
  value_infos[node_arg.Name()] = std::move(ep_value_info);
  return result;
}

// Static class function to create a std::unique_ptr<EpGraph>.
std::unique_ptr<EpGraph> EpGraph::Create(const GraphViewer& graph_viewer) {
  auto ep_graph = std::make_unique<EpGraph>(graph_viewer, PrivateTag{});

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  InlinedVector<EpValueInfo*> graph_inputs;
  InlinedVector<EpValueInfo*> graph_outputs;

  graph_inputs.reserve(graph_viewer.GetInputs().size());
  graph_outputs.reserve(graph_viewer.GetOutputs().size());

  for (const NodeArg* graph_input_node_arg : graph_viewer.GetInputs()) {
    assert(graph_input_node_arg != nullptr);
    graph_inputs.push_back(AddValueInfo(value_infos, *graph_input_node_arg, *ep_graph));
  }

  for (const NodeArg* graph_output_node_arg : graph_viewer.GetOutputs()) {
    assert(graph_output_node_arg != nullptr);
    graph_outputs.push_back(AddValueInfo(value_infos, *graph_output_node_arg, *ep_graph));
  }

  std::vector<std::unique_ptr<EpNode>> nodes;
  std::unordered_map<NodeIndex, EpNode*> index_to_node;

  nodes.reserve(graph_viewer.NumberOfNodes());
  index_to_node.reserve(graph_viewer.NumberOfNodes());

  for (const Node& node : graph_viewer.Nodes()) {
    InlinedVector<EpValueInfo*> node_inputs;
    node_inputs.reserve(node.InputDefs().size());

    for (const NodeArg* input : node.InputDefs()) {
      if (input != nullptr && input->Exists()) {
        node_inputs.push_back(AddValueInfo(value_infos, *input, *ep_graph));
      }
    }

    InlinedVector<EpValueInfo*> node_outputs;
    node_outputs.reserve(node.OutputDefs().size());

    for (const NodeArg* output : node.OutputDefs()) {
      if (output != nullptr && output->Exists()) {
        node_outputs.push_back(AddValueInfo(value_infos, *output, *ep_graph));
      }
    }

    auto ep_node = std::make_unique<EpNode>(node, std::move(node_inputs), std::move(node_outputs));
    nodes.push_back(std::move(ep_node));
    index_to_node[node.Index()] = nodes.back().get();
  }

  ep_graph->nodes = std::move(nodes);
  ep_graph->index_to_node = std::move(index_to_node);
  ep_graph->value_infos = std::move(value_infos);
  ep_graph->inputs = std::move(graph_inputs);
  ep_graph->outputs = std::move(graph_outputs);
  return ep_graph;
}

const std::string& EpGraph::Name() const { return graph_viewer.Name(); }

size_t EpGraph::NumberOfNodes() const { return nodes.size(); }

std::vector<const OrtNode*> EpGraph::GetNodes(int order) const {
  ExecutionOrder execution_order = static_cast<ExecutionOrder>(order);
  const std::vector<NodeIndex>& node_indices = graph_viewer.GetNodesInTopologicalOrder(execution_order);

  std::vector<const OrtNode*> result;
  result.reserve(NumberOfNodes());

  for (NodeIndex node_idx : node_indices) {
    auto node_it = index_to_node.find(node_idx);
    ORT_ENFORCE(node_it != index_to_node.end());
    result.push_back(node_it->second->ToExternal());
  }
  return result;
}

}  // namespace onnxruntime
