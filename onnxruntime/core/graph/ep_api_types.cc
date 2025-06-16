// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ep_api_types.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"

namespace onnxruntime {

// Create an EpValueInfo from a NodeArg.
static std::unique_ptr<EpValueInfo> CreateValueInfo(const NodeArg& node_arg, const EpGraph* ep_graph, size_t flags) {
  const auto* type_proto = node_arg.TypeAsProto();
  std::unique_ptr<OrtTypeInfo> type_info = type_proto != nullptr ? OrtTypeInfo::FromTypeProto(*type_proto)
                                                                 : nullptr;
  return std::make_unique<EpValueInfo>(ep_graph, node_arg.Name(), std::move(type_info), flags);
}

// Convert an array of NodeArgs to an array of EpValueInfos. The value_infos array should be the same size as the
// array of NodeArgs before calling this function.
static void ConvertNodeArgsToValueInfos(const EpGraph* ep_graph,
                                        std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos_map,
                                        gsl::span<const NodeArg* const> node_args, gsl::span<EpValueInfo*> value_infos,
                                        EpValueInfo::Flags value_flag) {
  assert(node_args.size() == value_infos.size());

  for (size_t i = 0; i < node_args.size(); ++i) {
    gsl::not_null<const NodeArg*> node_arg = node_args[i];
    const std::string& value_name = node_arg->Name();

    if (!node_arg->Exists()) {
      // A missing optional input/output has a null OrtValueInfo.
      value_infos[i] = nullptr;
      continue;
    }

    auto value_info_iter = value_infos_map.find(value_name);

    if (value_info_iter != value_infos_map.end()) {
      EpValueInfo* value_info = value_info_iter->second.get();
      value_info->SetFlag(value_flag);
      value_infos[i] = value_info;
    } else {
      std::unique_ptr<EpValueInfo> value_info = CreateValueInfo(*node_arg, ep_graph, value_flag);
      value_infos[i] = value_info.get();
      value_infos_map.emplace(value_name, std::move(value_info));
    }
  }
}

//
// EpNode
//

EpNode::EpNode(const EpGraph* ep_graph, const Node& node, PrivateTag)
    : OrtNode(OrtGraphIrApi::kEpApi), ep_graph(ep_graph), node(node) {}

Status EpNode::Create(const Node& node, const EpGraph* ep_graph,
                      std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos_map,
                      /*out*/ std::unique_ptr<EpNode>& result) {
  auto ep_node = std::make_unique<EpNode>(ep_graph, node, PrivateTag{});

  auto node_inputs = node.InputDefs();
  InlinedVector<EpValueInfo*> ep_node_inputs(node_inputs.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_inputs, ep_node_inputs, EpValueInfo::kFlagNone);

  auto node_outputs = node.OutputDefs();
  InlinedVector<EpValueInfo*> ep_node_outputs(node_outputs.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_outputs, ep_node_outputs, EpValueInfo::kFlagNone);

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
    const auto node_implicit_inputs = node.ImplicitInputDefs();
    ep_node_implicit_inputs.resize(node_implicit_inputs.size(), nullptr);
    ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_implicit_inputs, ep_node_implicit_inputs,
                                EpValueInfo::kFlagNone);

    std::vector<gsl::not_null<const Graph*>> node_subgraphs = node.GetSubgraphs();
    ep_node_subgraphs.reserve(node_subgraphs.size());
    for (gsl::not_null<const Graph*> subgraph : node_subgraphs) {
      SubgraphState subgraph_state;
      subgraph_state.subgraph_viewer = std::make_unique<GraphViewer>(*subgraph);
      ORT_RETURN_IF_ERROR(EpGraph::Create(*subgraph_state.subgraph_viewer, ep_node.get(), subgraph_state.ep_subgraph));
      ep_node_subgraphs.emplace_back(std::move(subgraph_state));
    }
  }

  ep_node->inputs = std::move(ep_node_inputs);
  ep_node->outputs = std::move(ep_node_outputs);
  ep_node->attributes_map = std::move(ep_node_attributes_map);
  ep_node->attributes = std::move(ep_node_attributes);
  ep_node->implicit_inputs = std::move(ep_node_implicit_inputs);
  ep_node->subgraphs = std::move(ep_node_subgraphs);
  result = std::move(ep_node);
  return Status::OK();
}

size_t EpNode::Id() const { return node.Index(); }
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

//
// EpValueInfo
//
EpValueInfo::EpValueInfo(const EpGraph* graph, const std::string& name, std::unique_ptr<OrtTypeInfo>&& type_info,
                         size_t flags)
    : OrtValueInfo(OrtGraphIrApi::kEpApi),
      graph_(graph),
      name_(name),
      type_info_(std::move(type_info)),
      flags_(flags) {}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
static Status GetInputIndices(const EpNode& consumer_node,
                              const std::string& value_info_name,
                              /*out*/ std::vector<int64_t>& indices) {
  bool found = false;
  auto add_input_indices =
      [&found, &value_info_name, &indices](gsl::span<const EpValueInfo* const> input_value_infos,
                                           bool is_implicit) -> void {
    for (size_t i = 0; i < input_value_infos.size(); i++) {
      if (input_value_infos[i]->Name() == value_info_name) {
        indices.push_back(is_implicit ? -1 : static_cast<int64_t>(i));
        found = true;
      }
    }
  };

  add_input_indices(consumer_node.inputs, false);
  add_input_indices(consumer_node.implicit_inputs, true);

  ORT_RETURN_IF_NOT(found, "Did not find OrtValueInfo with name ", value_info_name);
  return Status::OK();
}

static Status GetOutputIndex(const EpNode& producer_node,
                             const std::string& value_info_name,
                             /*out*/ size_t& index) {
  bool found = false;
  for (size_t i = 0; i < producer_node.outputs.size(); i++) {
    if (producer_node.outputs[i]->Name() == value_info_name) {
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

  if (graph_ == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get producer node for OrtValueInfo '", name_,
                           "' that is not owned by a OrtGraph.");
  }

  const Node* node = graph_->graph_viewer.GetProducerNode(name_);
  if (node == nullptr) {
    return Status::OK();
  }

  const EpNode* ep_node = graph_->index_to_ep_node.GetEpNode(node->Index());
  if (ep_node == nullptr) {
    return Status::OK();  // Node is not in this GraphViewer
  }

  size_t output_index = 0;
  ORT_RETURN_IF_ERROR(GetOutputIndex(*ep_node, name_, output_index));

  producer_info.node = ep_node->ToExternal();
  producer_info.output_index = output_index;
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(producer_info);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting the producer of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetConsumerInfos(std::vector<OrtValueInfo::ConsumerInfo>& result) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (graph_ == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get uses of OrtValueInfo '", name_,
                           "' that is not owned by a OrtGraph.");
  }

  std::vector<const Node*> nodes = graph_->graph_viewer.GetConsumerNodes(name_);
  if (nodes.empty()) {
    return Status::OK();
  }

  std::vector<OrtValueInfo::ConsumerInfo> consumer_infos;
  consumer_infos.reserve(nodes.size());
  for (const Node* node : nodes) {
    const EpNode* ep_node = graph_->index_to_ep_node.GetEpNode(node->Index());
    if (ep_node == nullptr) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<int64_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputIndices(*ep_node, name_, input_indices));

    for (int64_t input_index : input_indices) {
      OrtValueInfo::ConsumerInfo use_info(ep_node->ToExternal(), input_index);
      consumer_infos.push_back(use_info);
    }
  }

  result = std::move(consumer_infos);
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(result);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting the consumers of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetNumConsumerInfos(size_t& num_consumers) const {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  num_consumers = 0;

  if (graph_ == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to get number of uses of OrtValueInfo '", name_,
                           "' that is not owned by a OrtGraph.");
  }

  std::vector<const Node*> nodes = graph_->graph_viewer.GetConsumerNodes(name_);
  if (nodes.empty()) {
    return Status::OK();
  }

  for (const Node* node : nodes) {
    const EpNode* ep_node = graph_->index_to_ep_node.GetEpNode(node->Index());
    if (ep_node == nullptr) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<int64_t> input_indices;
    ORT_RETURN_IF_ERROR(GetInputIndices(*ep_node, name_, input_indices));

    num_consumers += input_indices.size();  // A single OrtNode can use an OrtValueInfo as an input more than once.
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(num_consumers);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Getting the consumers of an OrtValueInfo is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}

Status EpValueInfo::GetInitializerValue(const OrtValue*& result) const {
  if (!IsFlagSet(kIsInitializer)) {
    // This OrtValueInfo does not represent an initializer. Set result to nullptr and return an OK status
    // to allow user to use this function to check if this is an initializer.
    result = nullptr;
    return Status::OK();
  }

  const EpGraph* curr_graph = graph_;

  while (curr_graph != nullptr) {
    auto map_iter = curr_graph->initializer_values.find(name_);

    if (map_iter != curr_graph->initializer_values.end()) {
      result = map_iter->second.get();
      return Status::OK();
    }

    const EpNode* parent_node = curr_graph->parent_node;
    curr_graph = parent_node != nullptr ? parent_node->ep_graph : nullptr;
  }

  result = nullptr;
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to find initializer value named '", name_, "'.");
}

//
// EpGraph
//

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

static ONNX_NAMESPACE::TypeProto TypeProtoFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor) {
  ONNX_NAMESPACE::TypeProto t;
  t.mutable_tensor_type()->set_elem_type(tensor.data_type());
  auto shape = t.mutable_tensor_type()->mutable_shape();
  for (auto dim : tensor.dims())
    shape->add_dim()->set_dim_value(dim);

  return t;
}

// Static class function to create a std::unique_ptr<EpGraph>.
Status EpGraph::Create(const GraphViewer& graph_viewer, const EpNode* parent_ep_node,
                       /*out*/ std::unique_ptr<EpGraph>& result) {
  auto ep_graph = std::make_unique<EpGraph>(graph_viewer, parent_ep_node, PrivateTag{});

  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos_map;

  // Process graph inputs.
  const std::vector<const NodeArg*>& graph_input_node_args = graph_viewer.GetInputsIncludingInitializers();
  InlinedVector<EpValueInfo*> graph_input_value_infos(graph_input_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_input_node_args, graph_input_value_infos,
                              EpValueInfo::kIsGraphInput);

  // Process graph outputs.
  const std::vector<const NodeArg*>& graph_output_node_args = graph_viewer.GetOutputs();
  InlinedVector<EpValueInfo*> graph_output_value_infos(graph_output_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_output_node_args, graph_output_value_infos,
                              EpValueInfo::kIsGraphOutput);

  // If this is a subgraph (has parent node), add the value infos that come from the outer scope.
  if (parent_ep_node != nullptr) {
    const auto parent_implicit_inputs = parent_ep_node->node.ImplicitInputDefs();
    for (gsl::not_null<const NodeArg*> implicit_node_arg : parent_implicit_inputs) {
      const std::string& implicit_name = implicit_node_arg->Name();
      const GraphViewer& parent_graph_viewer = parent_ep_node->ep_graph->graph_viewer;
      bool is_outer_scope_initializer = parent_graph_viewer.GetGraph().GetInitializer(implicit_name, true) != nullptr;
      size_t flags = EpValueInfo::kIsOuterScope;

      if (is_outer_scope_initializer) {
        flags |= EpValueInfo::kIsInitializer;
      }

      assert(value_infos_map.count(implicit_name) == 0);  // Should not have added these yet.
      value_infos_map.emplace(implicit_node_arg->Name(), CreateValueInfo(*implicit_node_arg, ep_graph.get(), flags));
    }
  }

  // Create OrtValueInfo and OrtValue instances for each initializer.
  const InitializedTensorSet initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<EpValueInfo*> initializer_value_infos;
  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> initializer_values;
  AllocatorPtr initializer_allocator = CPUAllocator::DefaultInstance();

  initializer_value_infos.reserve(initializers.size());
  initializer_values.reserve(initializers.size());

  for (const auto& [initializer_name, tensor_proto] : initializers) {
    EpValueInfo* value_info = nullptr;

    auto iter = value_infos_map.find(initializer_name);
    if (iter != value_infos_map.end()) {
      value_info = iter->second.get();
      value_info->SetFlag(EpValueInfo::kIsInitializer);
    } else {
      auto type_proto = TypeProtoFromTensorProto(*tensor_proto);
      std::unique_ptr<OrtTypeInfo> type_info = OrtTypeInfo::FromTypeProto(type_proto);
      auto unique_value_info = std::make_unique<EpValueInfo>(ep_graph.get(), initializer_name, std::move(type_info),
                                                             EpValueInfo::kIsInitializer);

      value_info = unique_value_info.get();
      value_infos_map.emplace(initializer_name, std::move(unique_value_info));
    }

    initializer_value_infos.push_back(value_info);

    // Temporary: Copy onnx::TensorProto into OrtValue objects owned by this EpGraph.
    // TODO: Remove this logic once a separate PR that updates onnxruntime::Graph to store initializers as
    // OrtValue instances is merged.
    auto initializer_value = std::make_unique<OrtValue>();
    ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), graph_viewer.ModelPath(), *tensor_proto,
                                                     initializer_allocator, *initializer_value));
    initializer_values.emplace(value_info->Name(), std::move(initializer_value));
  }

  // Process nodes in topological order, converting Node to EpNode.
  std::vector<std::unique_ptr<EpNode>> ep_nodes;
  IndexToEpNodeMap index_to_ep_node;
  NodeIndex min_node_index = std::numeric_limits<NodeIndex>::max();
  NodeIndex max_node_index = std::numeric_limits<NodeIndex>::lowest();
  ep_nodes.reserve(graph_viewer.NumberOfNodes());

  const std::vector<NodeIndex>& node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);

  for (NodeIndex node_index : node_indices) {
    gsl::not_null<const Node*> node = graph_viewer.GetNode(node_index);
    std::unique_ptr<EpNode> ep_node = nullptr;

    ORT_RETURN_IF_ERROR(EpNode::Create(*node, ep_graph.get(), value_infos_map, ep_node));
    ep_nodes.push_back(std::move(ep_node));

    min_node_index = std::min(min_node_index, node->Index());
    max_node_index = std::max(max_node_index, node->Index());
  }

  // Iterate through nodes again and update the map of NodeIndex to EpNode*
  index_to_ep_node.Resize(min_node_index, max_node_index);
  for (std::unique_ptr<EpNode>& ep_node : ep_nodes) {
    index_to_ep_node.SetEpNode(ep_node->node.Index(), ep_node.get());
  }

  ep_graph->nodes = std::move(ep_nodes);
  ep_graph->index_to_ep_node = std::move(index_to_ep_node);
  ep_graph->value_infos = std::move(value_infos_map);
  ep_graph->initializer_value_infos = std::move(initializer_value_infos);
  ep_graph->initializer_values = std::move(initializer_values);
  ep_graph->inputs = std::move(graph_input_value_infos);
  ep_graph->outputs = std::move(graph_output_value_infos);
  result = std::move(ep_graph);
  return Status::OK();
}

const std::string& EpGraph::Name() const { return graph_viewer.Name(); }
int64_t EpGraph::OnnxIRVersion() const { return graph_viewer.GetOnnxIRVersion(); }
size_t EpGraph::NumInputs() const { return inputs.size(); }
size_t EpGraph::NumOutputs() const { return outputs.size(); }
size_t EpGraph::NumInitializers() const { return initializer_value_infos.size(); }

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

Status EpGraph::GetInitializers(std::vector<const OrtValueInfo*>& result) const {
  result.resize(initializer_value_infos.size());
  for (size_t i = 0; i < initializer_value_infos.size(); i++) {
    result[i] = initializer_value_infos[i];
  }
  return Status::OK();
}

size_t EpGraph::NumNodes() const { return nodes.size(); }

std::vector<const OrtNode*> EpGraph::GetNodes() const {
  std::vector<const OrtNode*> result;
  result.reserve(nodes.size());

  for (const std::unique_ptr<EpNode>& ep_node : nodes) {
    result.push_back(ep_node->ToExternal());
  }
  return result;
}

Status EpGraph::GetParentNode(const OrtNode*& result) const {
  result = parent_node != nullptr ? parent_node->ToExternal() : nullptr;
  return Status::OK();
}

}  // namespace onnxruntime
