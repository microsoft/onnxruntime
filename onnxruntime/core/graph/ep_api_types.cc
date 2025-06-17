// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ep_api_types.h"

#include <algorithm>
#include <cassert>
#include <functional>
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
                                        std::function<void(EpValueInfo*)> set_value_info_flags = nullptr) {
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

      if (set_value_info_flags) {
        set_value_info_flags(value_info);
      }

      value_infos[i] = value_info;
    } else {
      std::unique_ptr<EpValueInfo> value_info = CreateValueInfo(*node_arg, ep_graph, EpValueInfo::Flags::kFlagNone);

      if (set_value_info_flags) {
        set_value_info_flags(value_info.get());
      }

      value_infos[i] = value_info.get();
      value_infos_map.emplace(value_name, std::move(value_info));
    }
  }
}

//
// EpNode
//

EpNode::EpNode(const EpGraph* ep_graph, const Node& node, PrivateTag)
    : OrtNode(OrtGraphIrApi::kEpApi), ep_graph_(ep_graph), node_(node) {}

Status EpNode::Create(const Node& node, const EpGraph* ep_graph,
                      std::unordered_map<std::string, std::unique_ptr<EpValueInfo>>& value_infos_map,
                      /*out*/ std::unique_ptr<EpNode>& result) {
  auto ep_node = std::make_unique<EpNode>(ep_graph, node, PrivateTag{});

  auto node_inputs = node.InputDefs();
  auto node_outputs = node.OutputDefs();
  OrtConstPointerArray ep_node_inputs(ORT_TYPE_TAG_OrtValueInfo, node_inputs.size(), nullptr);
  OrtConstPointerArray ep_node_outputs(ORT_TYPE_TAG_OrtValueInfo, node_outputs.size(), nullptr);

  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_inputs, ep_node_inputs.ToSpan<EpValueInfo>());
  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_outputs, ep_node_outputs.ToSpan<EpValueInfo>());

  std::vector<SubgraphState> ep_node_subgraphs;
  OrtConstPointerArray ep_node_implicit_inputs(ORT_TYPE_TAG_OrtValueInfo);

  if (node.ContainsSubgraph()) {
    const auto node_implicit_inputs = node.ImplicitInputDefs();
    ep_node_implicit_inputs.Resize(node_implicit_inputs.size(), nullptr);

    ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_implicit_inputs,
                                ep_node_implicit_inputs.ToSpan<EpValueInfo>());

    std::vector<gsl::not_null<const Graph*>> node_subgraphs = node.GetSubgraphs();
    ep_node_subgraphs.reserve(node_subgraphs.size());

    for (gsl::not_null<const Graph*> subgraph : node_subgraphs) {
      SubgraphState subgraph_state;
      subgraph_state.subgraph_viewer = std::make_unique<GraphViewer>(*subgraph);
      ORT_RETURN_IF_ERROR(EpGraph::Create(*subgraph_state.subgraph_viewer, subgraph_state.ep_subgraph));
      subgraph_state.ep_subgraph->SetParentNode(ep_node.get());

      ep_node_subgraphs.emplace_back(std::move(subgraph_state));
    }
  }

  ep_node->inputs_ = std::move(ep_node_inputs);
  ep_node->outputs_ = std::move(ep_node_outputs);
  ep_node->implicit_inputs_ = std::move(ep_node_implicit_inputs);
  ep_node->subgraphs_ = std::move(ep_node_subgraphs);

  result = std::move(ep_node);

  return Status::OK();
}

size_t EpNode::GetId() const { return node_.Index(); }

const std::string& EpNode::GetName() const { return node_.Name(); }

const std::string& EpNode::GetOpType() const { return node_.OpType(); }

const std::string& EpNode::GetDomain() const { return node_.Domain(); }

Status EpNode::GetSinceVersion(int& since_version) const {
  since_version = node_.SinceVersion();
  return Status::OK();
}

Status EpNode::GetInputs(const OrtConstPointerArray*& result) const {
  result = &inputs_;
  return Status::OK();
}

Status EpNode::GetOutputs(const OrtConstPointerArray*& result) const {
  result = &outputs_;
  return Status::OK();
}

Status EpNode::GetImplicitInputs(const OrtConstPointerArray*& result) const {
  result = &implicit_inputs_;
  return Status::OK();
}

Status EpNode::GetNumSubgraphs(size_t& num_subgraphs) const {
  num_subgraphs = subgraphs_.size();
  return Status::OK();
}

Status EpNode::GetSubgraphs(InlinedVector<const OrtGraph*>& result) const {
  result.resize(subgraphs_.size());

  for (size_t i = 0; i < subgraphs_.size(); i++) {
    result[i] = subgraphs_[i].ep_subgraph->ToExternal();
  }

  return Status::OK();
}

Status EpNode::GetParentGraph(const OrtGraph*& parent_graph) const {
  parent_graph = ep_graph_->ToExternal();
  return Status::OK();
}

gsl::span<const EpValueInfo* const> EpNode::GetInputsSpan() const {
  return inputs_.ToConstSpan<EpValueInfo>();
}

gsl::span<const EpValueInfo* const> EpNode::GetImplicitInputsSpan() const {
  return implicit_inputs_.ToConstSpan<EpValueInfo>();
}

gsl::span<const EpValueInfo* const> EpNode::GetOutputsSpan() const {
  return outputs_.ToConstSpan<EpValueInfo>();
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
      if (input_value_infos[i]->GetName() == value_info_name) {
        indices.push_back(is_implicit ? -1 : static_cast<int64_t>(i));
        found = true;
      }
    }
  };

  add_input_indices(consumer_node.GetInputsSpan(), false);
  add_input_indices(consumer_node.GetImplicitInputsSpan(), true);

  ORT_RETURN_IF_NOT(found, "Did not find OrtValueInfo with name ", value_info_name);
  return Status::OK();
}

static Status GetOutputIndex(const EpNode& producer_node,
                             const std::string& value_info_name,
                             /*out*/ size_t& index) {
  bool found = false;
  gsl::span<const EpValueInfo* const> outputs = producer_node.GetOutputsSpan();

  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->GetName() == value_info_name) {
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

  const Node* node = graph_->GetGraphViewer().GetProducerNode(name_);
  if (node == nullptr) {
    return Status::OK();
  }

  const EpNode* ep_node = graph_->GetNode(node->Index());
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

  std::vector<const Node*> nodes = graph_->GetGraphViewer().GetConsumerNodes(name_);
  if (nodes.empty()) {
    return Status::OK();
  }

  std::vector<OrtValueInfo::ConsumerInfo> consumer_infos;
  consumer_infos.reserve(nodes.size());

  for (const Node* node : nodes) {
    const EpNode* ep_node = graph_->GetNode(node->Index());
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

  std::vector<const Node*> nodes = graph_->GetGraphViewer().GetConsumerNodes(name_);
  if (nodes.empty()) {
    return Status::OK();
  }

  for (const Node* node : nodes) {
    const EpNode* ep_node = graph_->GetNode(node->Index());
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
  if (!IsFlagSet(kIsConstantInitializer) && !IsFlagSet(kIsOptionalGraphInput)) {
    // This OrtValueInfo does not represent an initializer. Set result to nullptr and return an OK status
    // to allow user to use this function to check if this is an initializer.
    result = nullptr;
    return Status::OK();
  }

  ORT_RETURN_IF(graph_ == nullptr, "Unable to get initializer value named '", name_, "': parent graph is NULL");

  // This gets an initializer value defined in this graph or in a parent graph (as long as the value
  // is used in this graph).
  result = graph_->GetInitializerValue(name_);
  ORT_RETURN_IF(result == nullptr, "Unable to find initializer value named '", name_, "'.");
  return Status::OK();
}

Status EpValueInfo::IsRequiredGraphInput(bool& is_required_graph_input) const {
  is_required_graph_input = IsFlagSet(Flags::kIsRequiredGraphInput);
  return Status::OK();
}

Status EpValueInfo::IsOptionalGraphInput(bool& is_optional_graph_input) const {
  is_optional_graph_input = IsFlagSet(Flags::kIsOptionalGraphInput);
  return Status::OK();
}

Status EpValueInfo::IsGraphOutput(bool& is_graph_output) const {
  is_graph_output = IsFlagSet(Flags::kIsGraphOutput);
  return Status::OK();
}

Status EpValueInfo::IsConstantInitializer(bool& is_const_initializer) const {
  is_const_initializer = IsFlagSet(Flags::kIsConstantInitializer);
  return Status::OK();
}

Status EpValueInfo::IsFromOuterScope(bool& is_outer_scope) const {
  is_outer_scope = IsFlagSet(Flags::kIsOuterScope);
  return Status::OK();
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

EpGraph::EpGraph(const GraphViewer& graph_viewer, PrivateTag)
    : OrtGraph(OrtGraphIrApi::kEpApi), graph_viewer_(graph_viewer) {}

// Static class function to create a std::unique_ptr<EpGraph>.
Status EpGraph::Create(const GraphViewer& graph_viewer, /*out*/ std::unique_ptr<EpGraph>& result) {
  auto ep_graph = std::make_unique<EpGraph>(graph_viewer, PrivateTag{});

  AllocatorPtr initializer_allocator = CPUAllocator::DefaultInstance();
  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos_map;

  // Process graph inputs.
  const std::vector<const NodeArg*>& graph_input_node_args = graph_viewer.GetInputsIncludingInitializers();
  OrtConstPointerArray graph_input_value_infos(ORT_TYPE_TAG_OrtValueInfo, graph_input_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_input_node_args,
                              graph_input_value_infos.ToSpan<EpValueInfo>(),
                              [&graph_viewer](EpValueInfo* v) {
                                if (!graph_viewer.IsInitializedTensor(v->GetName())) {
                                  v->SetFlag(EpValueInfo::Flags::kIsRequiredGraphInput);
                                } else if (graph_viewer.CanOverrideInitializer()) {
                                  v->SetFlag(EpValueInfo::Flags::kIsOptionalGraphInput);
                                }
                              });

  // Process graph outputs.
  const std::vector<const NodeArg*>& graph_output_node_args = graph_viewer.GetOutputs();
  OrtConstPointerArray graph_output_value_infos(ORT_TYPE_TAG_OrtValueInfo, graph_output_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_output_node_args,
                              graph_output_value_infos.ToSpan<EpValueInfo>(),
                              [](EpValueInfo* v) { v->SetFlag(EpValueInfo::Flags::kIsGraphOutput); });

  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> outer_scope_initializer_values;

  // Create OrtValueInfo and OrtValue instances for each initializer.
  const InitializedTensorSet initializers = graph_viewer.GetAllInitializedTensors();
  OrtConstPointerArray initializer_value_infos(ORT_TYPE_TAG_OrtValueInfo);
  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> initializer_values;

  initializer_value_infos.storage.reserve(initializers.size());
  initializer_values.reserve(initializers.size());

  for (const auto& [initializer_name, tensor_proto] : initializers) {
    EpValueInfo* value_info = nullptr;
    EpValueInfo::Flags flag = graph_viewer.IsConstantInitializer(initializer_name, /*check_outer_scope*/ false)
                                  ? EpValueInfo::kIsConstantInitializer
                                  : EpValueInfo::kIsOptionalGraphInput;

    auto iter = value_infos_map.find(initializer_name);
    if (iter != value_infos_map.end()) {
      value_info = iter->second.get();
      value_info->SetFlag(flag);
    } else {
      auto type_proto = utils::TypeProtoFromTensorProto(*tensor_proto);
      std::unique_ptr<OrtTypeInfo> type_info = OrtTypeInfo::FromTypeProto(type_proto);
      auto unique_value_info = std::make_unique<EpValueInfo>(ep_graph.get(), initializer_name, std::move(type_info),
                                                             flag);

      value_info = unique_value_info.get();
      value_infos_map.emplace(initializer_name, std::move(unique_value_info));
    }

    initializer_value_infos.storage.push_back(value_info);

    // Temporary: Copy onnx::TensorProto into OrtValue objects owned by this EpGraph.
    // TODO: Remove this logic once a separate PR that updates onnxruntime::Graph to store initializers as
    // OrtValue instances is merged.
    auto initializer_value = std::make_unique<OrtValue>();
    ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), graph_viewer.ModelPath(), *tensor_proto,
                                                     initializer_allocator, *initializer_value));
    initializer_values.emplace(value_info->GetName(), std::move(initializer_value));
  }

  // Process nodes in topological order, converting Node to EpNode.
  std::vector<std::unique_ptr<EpNode>> ep_nodes_holder;
  OrtConstPointerArray ep_nodes(ORT_TYPE_TAG_OrtNode);
  IndexToEpNodeMap index_to_ep_node;
  NodeIndex min_node_index = std::numeric_limits<NodeIndex>::max();
  NodeIndex max_node_index = std::numeric_limits<NodeIndex>::lowest();

  ep_nodes_holder.reserve(graph_viewer.NumberOfNodes());
  ep_nodes.storage.reserve(graph_viewer.NumberOfNodes());

  const std::vector<NodeIndex>& node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);

  for (NodeIndex node_index : node_indices) {
    gsl::not_null<const Node*> node = graph_viewer.GetNode(node_index);
    std::unique_ptr<EpNode> ep_node = nullptr;

    ORT_RETURN_IF_ERROR(EpNode::Create(*node, ep_graph.get(), value_infos_map, ep_node));

    ep_nodes.storage.push_back(ep_node.get());
    ep_nodes_holder.push_back(std::move(ep_node));

    min_node_index = std::min(min_node_index, node->Index());
    max_node_index = std::max(max_node_index, node->Index());
  }

  // Iterate through nodes again and update the map of NodeIndex to EpNode*
  index_to_ep_node.Resize(min_node_index, max_node_index);
  for (std::unique_ptr<EpNode>& ep_node : ep_nodes_holder) {
    index_to_ep_node.SetEpNode(ep_node->GetInternalNode().Index(), ep_node.get());
  }

  // If this is a subgraph, add the OrtValueInfo and OrtValue objects that come from the outer scope.
  // Wait until we have already processed OrtValueInfos consumed and produced by nodes so that we only add
  // outer OrtValueInfo/OrtValue if they are actually used by the nodes in this GraphViewer.
  if (graph_viewer.IsSubgraph()) {
    gsl::not_null<const Graph*> parent_graph = graph_viewer.GetGraph().ParentGraph();
    gsl::not_null<const Node*> parent_node = graph_viewer.ParentNode();

    for (gsl::not_null<const NodeArg*> implicit_node_arg : parent_node->ImplicitInputDefs()) {
      const std::string& implicit_name = implicit_node_arg->Name();
      auto value_info_iter = value_infos_map.find(implicit_name);

      if (value_info_iter == value_infos_map.end()) {
        continue;  // Skip. This implicit value is not used by a node in this GraphViewer.
      }

      EpValueInfo* outer_value_info = value_info_iter->second.get();
      bool is_constant = false;
      const ONNX_NAMESPACE::TensorProto* outer_initializer = parent_graph->GetInitializer(implicit_name,
                                                                                          /*check_outer_scope*/ true,
                                                                                          is_constant);
      outer_value_info->SetFlag(EpValueInfo::kIsOuterScope);

      if (outer_initializer != nullptr) {
        outer_value_info->SetFlag(is_constant ? EpValueInfo::kIsConstantInitializer : EpValueInfo::kIsOptionalGraphInput);
      }

      // Temporary: Copy onnx::TensorProto into OrtValue objects owned by this EpGraph.
      // TODO: Remove this logic once a separate PR that updates onnxruntime::Graph to store initializers as
      // OrtValue instances is merged.
      if (outer_initializer != nullptr) {
        auto initializer_value = std::make_unique<OrtValue>();
        ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), parent_graph->ModelPath(),
                                                         *outer_initializer, initializer_allocator,
                                                         *initializer_value));
        outer_scope_initializer_values.emplace(outer_value_info->GetName(), std::move(initializer_value));
      }
    }
  }

  ep_graph->nodes_holder_ = std::move(ep_nodes_holder);
  ep_graph->nodes_ = std::move(ep_nodes);
  ep_graph->index_to_ep_node_ = std::move(index_to_ep_node);
  ep_graph->value_infos_ = std::move(value_infos_map);
  ep_graph->initializer_value_infos_ = std::move(initializer_value_infos);
  ep_graph->initializer_values_ = std::move(initializer_values);
  ep_graph->outer_scope_initializer_values_ = std::move(outer_scope_initializer_values);
  ep_graph->inputs_ = std::move(graph_input_value_infos);
  ep_graph->outputs_ = std::move(graph_output_value_infos);

  result = std::move(ep_graph);

  return Status::OK();
}

const std::string& EpGraph::GetName() const { return graph_viewer_.Name(); }

int64_t EpGraph::GetOnnxIRVersion() const { return graph_viewer_.GetOnnxIRVersion(); }

Status EpGraph::GetInputs(const OrtConstPointerArray*& result) const {
  result = &inputs_;
  return Status::OK();
}

Status EpGraph::GetOutputs(const OrtConstPointerArray*& result) const {
  result = &outputs_;
  return Status::OK();
}

Status EpGraph::GetInitializers(const OrtConstPointerArray*& result) const {
  result = &initializer_value_infos_;
  return Status::OK();
}

Status EpGraph::GetNodes(const OrtConstPointerArray*& result) const {
  result = &nodes_;
  return Status::OK();
}

Status EpGraph::GetParentNode(const OrtNode*& result) const {
  result = parent_node_ != nullptr ? parent_node_->ToExternal() : nullptr;
  return Status::OK();
}

void EpGraph::SetParentNode(const EpNode* node) { parent_node_ = node; }

const GraphViewer& EpGraph::GetGraphViewer() const { return graph_viewer_; }

const EpNode* EpGraph::GetNode(NodeIndex node_index) const {
  return index_to_ep_node_.GetEpNode(node_index);
}

const OrtValue* EpGraph::GetInitializerValue(std::string_view name) const {
  // Check for initializer value in the graph's scope.
  if (auto iter = initializer_values_.find(name);
      iter != initializer_values_.end()) {
    return iter->second.get();
  }

  // Check for the initializer value in an outer scope.
  // Only finds a value if the outer initializer value is used within this graph.
  if (auto iter = outer_scope_initializer_values_.find(name);
      iter != outer_scope_initializer_values_.end()) {
    return iter->second.get();
  }

  return nullptr;
}
}  // namespace onnxruntime
