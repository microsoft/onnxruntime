// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ep_api_types.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"

namespace onnxruntime {

static EpValueInfo* AddValueInfo(std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos,
                                 const NodeArg& node_arg, const EpGraph* ep_graph) {
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

EpNode::EpNode(const EpGraph* ep_graph, const Node& node, PrivateTag)
    : OrtNode(OrtGraphIrApi::kEpApi), ep_graph(ep_graph), node(node) {}

std::unique_ptr<EpNode> EpNode::Create(const Node& node, const EpGraph* ep_graph,
                                       std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos_map) {
  auto init_ep_node_io = [&ep_graph, &value_infos_map](ConstPointerContainer<std::vector<NodeArg*>> node_args,
                                                       gsl::span<EpValueInfo*> value_infos) {
    assert(node_args.size() == value_infos.size());
    for (size_t i = 0; i < node_args.size(); i++) {
      const NodeArg* node_arg = node_args[i];
      assert(node_arg != nullptr);

      // Note that a missing optional input/output is assigned a null OrtValueInfo.
      value_infos[i] = node_arg->Exists() ? AddValueInfo(value_infos_map, *node_arg, ep_graph) : nullptr;
    }
  };
  auto ep_node = std::make_unique<EpNode>(ep_graph, node, PrivateTag{});

  auto node_inputs = node.InputDefs();
  InlinedVector<EpValueInfo*> ep_node_inputs(node_inputs.size(), nullptr);
  init_ep_node_io(node_inputs, ep_node_inputs);

  auto node_outputs = node.OutputDefs();
  InlinedVector<EpValueInfo*> ep_node_outputs(node_outputs.size(), nullptr);
  init_ep_node_io(node_outputs, ep_node_outputs);

  const auto& node_attrs = node.GetAttributes();
  std::unordered_map<std::string, std::unique_ptr<ONNX_NAMESPACE::AttributeProto>> ep_node_attributes_map;
  std::vector<OrtOpAttr*> ep_node_attributes;
  ep_node_attributes_map.reserve(node_attrs.size());
  ep_node_attributes.reserve(node_attrs.size());
  for (const auto& item : node_attrs) {
    auto attr = std::make_unique<ONNX_NAMESPACE::AttributeProto>(item.second);
    ep_node_attributes.emplace_back(reinterpret_cast<OrtOpAttr*>(attr.get()));
    ep_node_attributes_map[item.first] = std::move(attr);
  }

  std::vector<SubgraphState> ep_node_subgraphs;
  std::vector<EpValueInfo*> ep_node_implicit_inputs;

  if (node.ContainsSubgraph()) {
    auto node_implicit_inputs = node.ImplicitInputDefs();
    ep_node_implicit_inputs.resize(node_implicit_inputs.size(), nullptr);
    init_ep_node_io(node_implicit_inputs, ep_node_implicit_inputs);

    std::vector<gsl::not_null<const Graph*>> node_subgraphs = node.GetSubgraphs();
    ep_node_subgraphs.reserve(node_subgraphs.size());
    for (gsl::not_null<const Graph*> subgraph : node_subgraphs) {
      SubgraphState subgraph_state;
      subgraph_state.subgraph_viewer = std::make_unique<GraphViewer>(*subgraph);
      subgraph_state.ep_subgraph = EpGraph::Create(*subgraph_state.subgraph_viewer, ep_node.get());
      ep_node_subgraphs.emplace_back(std::move(subgraph_state));
    }
  }

  ep_node->inputs = std::move(ep_node_inputs);
  ep_node->outputs = std::move(ep_node_outputs);
  ep_node->attributes_map = std::move(ep_node_attributes_map);
  ep_node->attributes = std::move(ep_node_attributes);
  ep_node->implicit_inputs = std::move(ep_node_implicit_inputs);
  ep_node->subgraphs = std::move(ep_node_subgraphs);
  return ep_node;
}

const std::string& EpNode::Name() const { return node.Name(); }
const std::string& EpNode::OpType() const { return node.OpType(); }
const std::string& EpNode::Domain() const { return node.Domain(); }
Status EpNode::GetSinceVersion(int& since_version) const {
  since_version = node.SinceVersion();
  return Status::OK();
}
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

Status EpNode::GetNumImplicitInputs(size_t& num_implicit_inputs) const {
  num_implicit_inputs = implicit_inputs.size();
  return Status::OK();
}

Status EpNode::GetImplicitInputs(InlinedVector<const OrtValueInfo*>& result) const {
  result.resize(implicit_inputs.size());
  for (size_t i = 0; i < implicit_inputs.size(); i++) {
    result[i] = implicit_inputs[i];
  }
  return Status::OK();
}

Status EpNode::GetNumAttributes(size_t& num_attrs) const {
  num_attrs = attributes.size();
  return Status::OK();
}

Status EpNode::GetAttributes(onnxruntime::InlinedVector<const OrtOpAttr*>& result) const {
  result.resize(attributes.size());
  for (size_t i = 0; i < attributes.size(); i++) {
    result[i] = attributes[i];
  }
  return Status::OK();
}

Status EpNode::GetNumSubgraphs(size_t& num_subgraphs) const {
  num_subgraphs = subgraphs.size();
  return Status::OK();
}

Status EpNode::GetSubgraphs(InlinedVector<const OrtGraph*>& result) const {
  result.resize(subgraphs.size());
  for (size_t i = 0; i < subgraphs.size(); i++) {
    result[i] = subgraphs[i].ep_subgraph->ToExternal();
  }
  return Status::OK();
}

Status EpNode::GetParentGraph(const OrtGraph*& parent_graph) const {
  parent_graph = ep_graph->ToExternal();
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
static Status GetInputIndices(const EpNode& consumer_node,
                              const std::string& value_info_name,
                              /*out*/ std::vector<int64_t>& indices) {
  bool found = false;
  auto add_input_indices =
      [&found, &value_info_name, &indices](ConstPointerContainer<std::vector<NodeArg*>> input_defs,
                                           bool is_implicit) -> void {
    for (size_t i = 0; i < input_defs.size(); i++) {
      if (input_defs[i]->Name() == value_info_name) {
        indices.push_back(is_implicit ? -1 : static_cast<int64_t>(i));
        found = true;
      }
    }
  };

  const auto node_input_defs = consumer_node.node.InputDefs();
  indices.reserve(node_input_defs.size());
  add_input_indices(node_input_defs, false);

  if (!found) {
    // Check implicit inputs. Nodes that contain subgraphs (e.g., If, Loop) may have implicit inputs
    // that are consumed by nodes within their subgraph.
    add_input_indices(consumer_node.node.ImplicitInputDefs(), true);
  }

  ORT_RETURN_IF_NOT(found, "Did not find OrtValueInfo with name ", value_info_name);
  return Status::OK();
}

static Status GetOutputIndex(const EpNode& producer_node,
                             const std::string& value_info_name,
                             /*out*/ size_t& index) {
  bool found = false;
  for (size_t i = 0; i < producer_node.outputs.size(); i++) {
    if (producer_node.outputs[i]->name == value_info_name) {
      index = i;
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

  if (graph == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get producer node for OrtValueInfo '", name,
                           "' that is not owned by a OrtGraph.");
  }

  const Node* node = graph->graph_viewer.GetProducerNode(name);
  if (node == nullptr) {
    return Status::OK();
  }

  const EpNode* ep_node = graph->index_to_ep_node.GetEpNode(node->Index());
  if (ep_node == nullptr) {
    return Status::OK();  // Node is not in this GraphViewer
  }

  size_t output_index = 0;
  ORT_RETURN_IF_ERROR(GetOutputIndex(*ep_node, name, output_index));

  producer_info.node = ep_node->ToExternal();
  producer_info.output_index = output_index;
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(producer_info);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting producers from OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetConsumers(std::vector<OrtValueInfo::ConsumerInfo>& consumer_infos) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (graph == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get uses of OrtValueInfo '", name,
                           "' that is not owned by a OrtGraph.");
  }

  std::vector<const Node*> nodes = graph->graph_viewer.GetConsumerNodes(name);
  if (nodes.empty()) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(consumer_infos.empty(), "Internal error: consumer_infos should be empty in GetUses()");
  consumer_infos.reserve(nodes.size());
  for (const Node* node : nodes) {
    const EpNode* ep_node = graph->index_to_ep_node.GetEpNode(node->Index());
    if (ep_node == nullptr) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<int64_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputIndices(*ep_node, name, input_indices));

    for (int64_t input_index : input_indices) {
      OrtValueInfo::ConsumerInfo use_info(ep_node->ToExternal(), input_index);
      consumer_infos.push_back(use_info);
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(consumer_infos);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting uses of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetNumConsumers(size_t& num_consumers) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  num_consumers = 0;

  if (graph == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get number of uses of OrtValueInfo '", name,
                           "' that is not owned by a OrtGraph.");
  }

  std::vector<const Node*> nodes = graph->graph_viewer.GetConsumerNodes(name);
  if (nodes.empty()) {
    return Status::OK();
  }

  for (const Node* node : nodes) {
    const EpNode* ep_node = graph->index_to_ep_node.GetEpNode(node->Index());
    if (ep_node == nullptr) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<int64_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputIndices(*ep_node, name, input_indices));

    num_consumers += input_indices.size();  // A single OrtNode can use an OrtValueInfo as an input more than once.
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(num_consumers);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting uses of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

void EpGraph::IndexToEpNodeMap::Resize(NodeIndex min_node_index, NodeIndex max_node_index) {
  assert(max_node_index >= min_node_index);
  size_t num_elems = (max_node_index - min_node_index) + 1;

  min_node_index_ = min_node_index;
  nodes_.resize(num_elems, nullptr);
}

EpNode* EpGraph::IndexToEpNodeMap::GetEpNode(NodeIndex node_index) const {
  size_t i = node_index - min_node_index_;
  assert(i < nodes_.size());
  return nodes_[i];
}

void EpGraph::IndexToEpNodeMap::SetEpNode(NodeIndex node_index, EpNode* ep_node) {
  size_t i = node_index - min_node_index_;
  assert(i < nodes_.size());
  nodes_[i] = ep_node;
}

EpGraph::EpGraph(const GraphViewer& graph_viewer, const EpNode* parent_node, PrivateTag)
    : OrtGraph(OrtGraphIrApi::kEpApi), graph_viewer(graph_viewer), parent_node(parent_node) {}

// Static class function to create a std::unique_ptr<EpGraph>.
std::unique_ptr<EpGraph> EpGraph::Create(const GraphViewer& graph_viewer, const EpNode* parent_ep_node) {
  auto ep_graph = std::make_unique<EpGraph>(graph_viewer, parent_ep_node, PrivateTag{});

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  InlinedVector<EpValueInfo*> graph_inputs;
  InlinedVector<EpValueInfo*> graph_outputs;

  graph_inputs.reserve(graph_viewer.GetInputs().size());
  graph_outputs.reserve(graph_viewer.GetOutputs().size());

  for (const NodeArg* graph_input_node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    assert(graph_input_node_arg != nullptr);
    graph_inputs.push_back(AddValueInfo(value_infos, *graph_input_node_arg, ep_graph.get()));
  }

  for (const NodeArg* graph_output_node_arg : graph_viewer.GetOutputs()) {
    assert(graph_output_node_arg != nullptr);
    graph_outputs.push_back(AddValueInfo(value_infos, *graph_output_node_arg, ep_graph.get()));
  }

  std::vector<std::unique_ptr<EpNode>> ep_nodes;
  IndexToEpNodeMap index_to_ep_node;
  NodeIndex min_node_index = std::numeric_limits<NodeIndex>::max();
  NodeIndex max_node_index = std::numeric_limits<NodeIndex>::lowest();

  ep_nodes.reserve(graph_viewer.NumberOfNodes());
  for (const Node& node : graph_viewer.Nodes()) {
    ep_nodes.push_back(EpNode::Create(node, ep_graph.get(), value_infos));
    min_node_index = std::min(min_node_index, node.Index());
    max_node_index = std::max(max_node_index, node.Index());
  }

  index_to_ep_node.Resize(min_node_index, max_node_index);
  for (std::unique_ptr<EpNode>& ep_node : ep_nodes) {
    index_to_ep_node.SetEpNode(ep_node->node.Index(), ep_node.get());
  }

  ep_graph->nodes = std::move(ep_nodes);
  ep_graph->index_to_ep_node = std::move(index_to_ep_node);
  ep_graph->value_infos = std::move(value_infos);
  ep_graph->inputs = std::move(graph_inputs);
  ep_graph->outputs = std::move(graph_outputs);
  return ep_graph;
}

const std::string& EpGraph::Name() const { return graph_viewer.Name(); }
size_t EpGraph::NumInputs() const { return inputs.size(); }
size_t EpGraph::NumOutputs() const { return outputs.size(); }

Status EpGraph::GetInputs(InlinedVector<const OrtValueInfo*>& result) const {
  result.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    result[i] = inputs[i];
  }
  return Status::OK();
}

Status EpGraph::GetOutputs(InlinedVector<const OrtValueInfo*>& result) const {
  result.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    result[i] = outputs[i];
  }
  return Status::OK();
}

size_t EpGraph::NumNodes() const { return nodes.size(); }

std::vector<const OrtNode*> EpGraph::GetNodes(int order) const {
  ExecutionOrder execution_order = static_cast<ExecutionOrder>(order);
  const std::vector<NodeIndex>& node_indices = graph_viewer.GetNodesInTopologicalOrder(execution_order);

  std::vector<const OrtNode*> result;
  result.reserve(NumNodes());

  for (NodeIndex node_idx : node_indices) {
    const EpNode* ep_node = index_to_ep_node.GetEpNode(node_idx);
    assert(ep_node != nullptr);
    result.push_back(ep_node->ToExternal());
  }
  return result;
}

Status EpGraph::GetParentNode(const OrtNode*& result) const {
  result = parent_node != nullptr ? parent_node->ToExternal() : nullptr;
  return Status::OK();
}

}  // namespace onnxruntime
