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
#include "core/framework/tensor_external_data_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"

namespace onnxruntime {

template <typename DstElem>
static Status CheckCopyDestination(std::string_view error_array_label, size_t src_size, gsl::span<DstElem const> dst) {
  if (dst.size() < src_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Not enough space for ", error_array_label, ": expected buffer with room for at least ",
                           src_size, " elements, but got buffer with room for only ", dst.size(), " elements.");
  }

  if (dst.data() == nullptr && src_size > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer to store ", error_array_label, " is NULL.");
  }

  return Status::OK();
}

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

#if !defined(ORT_MINIMAL_BUILD)
static bool IsOptionalAttribute(const Node& node, const std::string& attr_name) {
  const ONNX_NAMESPACE::OpSchema* op_schema = node.Op();
  if (op_schema == nullptr) {
    return false;
  }

  auto attr_schema_iter = op_schema->attributes().find(attr_name);
  if (attr_schema_iter == op_schema->attributes().end()) {
    return false;  // Not an attribute for this operator type.
  }

  const ONNX_NAMESPACE::OpSchema::Attribute& attr_schema = attr_schema_iter->second;

  return !attr_schema.required;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

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
  InlinedVector<EpValueInfo*> ep_node_inputs(node_inputs.size(), nullptr);
  InlinedVector<EpValueInfo*> ep_node_outputs(node_outputs.size(), nullptr);

  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_inputs, ep_node_inputs);
  ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_outputs, ep_node_outputs);

  const auto& node_attrs = node.GetAttributes();
  std::unordered_map<std::string, std::unique_ptr<ONNX_NAMESPACE::AttributeProto>> ep_node_attributes_map;
  std::vector<OrtOpAttr*> ep_node_attributes;

  if (node_attrs.size() > 0) {
    ep_node_attributes.reserve(node_attrs.size());

    for (const auto& item : node_attrs) {
      auto attr = std::make_unique<ONNX_NAMESPACE::AttributeProto>(item.second);  // Copy AttributeProto and owned by this EpNode object.
      ep_node_attributes.push_back(reinterpret_cast<OrtOpAttr*>(attr.get()));
      ep_node_attributes_map.emplace(item.first, std::move(attr));
    }
  }

  std::vector<SubgraphState> ep_node_subgraphs;
  std::vector<EpValueInfo*> ep_node_implicit_inputs;

  if (node.ContainsSubgraph()) {
    const auto node_implicit_inputs = node.ImplicitInputDefs();
    ep_node_implicit_inputs.resize(node_implicit_inputs.size(), nullptr);

    ConvertNodeArgsToValueInfos(ep_graph, value_infos_map, node_implicit_inputs, ep_node_implicit_inputs);

    std::unordered_map<std::string, gsl::not_null<const Graph*>> subgraphs_map = node.GetAttributeNameToSubgraphMap();
    ep_node_subgraphs.reserve(subgraphs_map.size());

    for (const auto& [attr_name, subgraph] : subgraphs_map) {
      SubgraphState subgraph_state;
      subgraph_state.attribute_name = attr_name;
      subgraph_state.subgraph_viewer = std::make_unique<GraphViewer>(*subgraph);
      ORT_RETURN_IF_ERROR(EpGraph::Create(*subgraph_state.subgraph_viewer, subgraph_state.ep_subgraph));
      subgraph_state.ep_subgraph->SetParentNode(ep_node.get());

      ep_node_subgraphs.emplace_back(std::move(subgraph_state));
    }
  }

  ep_node->inputs_ = std::move(ep_node_inputs);
  ep_node->outputs_ = std::move(ep_node_outputs);
  ep_node->attributes_map_ = std::move(ep_node_attributes_map);
  ep_node->attributes_ = std::move(ep_node_attributes);
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

size_t EpNode::GetNumInputs() const {
  return inputs_.size();
}

Status EpNode::GetInputs(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_inputs = inputs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("node inputs", num_inputs, dst)));

  for (size_t i = 0; i < num_inputs; ++i) {
    dst[i] = inputs_[i];
  }

  return Status::OK();
}

size_t EpNode::GetNumOutputs() const {
  return outputs_.size();
}

Status EpNode::GetOutputs(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_outputs = outputs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("node outputs", num_outputs, dst)));

  for (size_t i = 0; i < num_outputs; ++i) {
    dst[i] = outputs_[i];
  }

  return Status::OK();
}

Status EpNode::GetNumImplicitInputs(size_t& num_implicit_inputs) const {
  num_implicit_inputs = implicit_inputs_.size();
  return Status::OK();
}

Status EpNode::GetImplicitInputs(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_implicit_inputs = implicit_inputs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("node implicit inputs", num_implicit_inputs, dst)));

  for (size_t i = 0; i < num_implicit_inputs; ++i) {
    dst[i] = implicit_inputs_[i];
  }

  return Status::OK();
}

size_t EpNode::GetNumAttributes() const {
  return attributes_.size();
}

Status EpNode::GetAttributes(gsl::span<const OrtOpAttr*> dst) const {
  const size_t num_attributes = attributes_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtOpAttr*>("node attributes", num_attributes, dst)));

  for (size_t i = 0; i < num_attributes; ++i) {
    dst[i] = attributes_[i];
  }

  return Status::OK();
}

Status EpNode::GetTensorAttributeAsOrtValue(const OrtOpAttr* attribute, OrtValue*& result) const {
  const auto* attr_proto = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attribute);

  if (attr_proto->type() != onnx::AttributeProto::TENSOR) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This OrtOpAttr instance is not a 'TENSOR' attribute");
  }

  const auto& graph_viewer = ep_graph_->GetGraphViewer();
  const auto& tensor_proto = attr_proto->t();

  // Check that TensorProto is valid.
  ORT_ENFORCE(utils::HasDataType(tensor_proto), "Tensor proto doesn't have data type.");
  ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(tensor_proto.data_type()), "Tensor proto has invalid data type.");
  ORT_ENFORCE(!utils::HasExternalData(tensor_proto),
              "Tensor proto with external data for value attribute is not supported.");

  // Initialize OrtValue for tensor attribute.
  auto tensor_attribute_value = std::make_unique<OrtValue>();
  AllocatorPtr tensor_attribute_allocator = CPUAllocator::DefaultInstance();
  ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), graph_viewer.ModelPath(), tensor_proto,
                                                   tensor_attribute_allocator, *tensor_attribute_value));

  result = tensor_attribute_value.release();
  return Status::OK();
}

Status EpNode::GetNumSubgraphs(size_t& num_subgraphs) const {
  num_subgraphs = subgraphs_.size();
  return Status::OK();
}

Status EpNode::GetSubgraphs(gsl::span<const OrtGraph*> subgraphs,
                            const char** opt_attribute_names) const {
  const size_t num_subgraphs = subgraphs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtGraph*>("node subgraphs", num_subgraphs, subgraphs)));

  for (size_t i = 0; i < num_subgraphs; ++i) {
    subgraphs[i] = subgraphs_[i].ep_subgraph.get();

    if (opt_attribute_names) {
      opt_attribute_names[i] = subgraphs_[i].attribute_name.c_str();
    }
  }

  return Status::OK();
}

Status EpNode::GetGraph(const OrtGraph*& parent_graph) const {
  parent_graph = ep_graph_->ToExternal();
  return Status::OK();
}

gsl::span<const EpValueInfo* const> EpNode::GetInputsSpan() const {
  return inputs_;
}

gsl::span<const EpValueInfo* const> EpNode::GetImplicitInputsSpan() const {
  return implicit_inputs_;
}

gsl::span<const EpValueInfo* const> EpNode::GetOutputsSpan() const {
  return outputs_;
}

const OrtOpAttr* EpNode::GetAttribute(const std::string& name, bool& is_unset_optional_attr) const {
  auto iter = attributes_map_.find(name);
  if (iter != attributes_map_.end()) {
    is_unset_optional_attr = false;
    return reinterpret_cast<const OrtOpAttr*>(iter->second.get());
  }

#if !defined(ORT_MINIMAL_BUILD)
  is_unset_optional_attr = IsOptionalAttribute(node_, name);
#else
  // This is not properly set in a minimal build because it does not have access to the operator schema.
  is_unset_optional_attr = false;
#endif  // !defined(ORT_MINIMAL_BUILD)
  return nullptr;
}

const std::string& EpNode::GetEpName() const {
  return node_.GetExecutionProviderType();
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
  ORT_RETURN_IF_ERROR(graph_->GetInitializerValue(name_, result));
  ORT_RETURN_IF(result == nullptr, "Unable to find initializer value named '", name_, "'.");
  return Status::OK();
}

Status EpValueInfo::GetExternalInitializerInfo(std::unique_ptr<ExternalDataInfo>& result) const {
  if (!IsFlagSet(kIsConstantInitializer) && !IsFlagSet(kIsOptionalGraphInput)) {
    result = nullptr;
    return Status::OK();
  }

  ORT_RETURN_IF(graph_ == nullptr, "Unable to get external initializer information for value named '",
                name_, "': parent graph is NULL");

  const onnxruntime::Graph& graph = graph_->GetGraphViewer().GetGraph();

  if (!graph.GetExternalInitializerInfo(name_, result, /*check_outer_scope*/ true)) {
    result = nullptr;
  }

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

EpGraph::EpGraph(std::unique_ptr<GraphViewer> graph_viewer,
                 std::unique_ptr<IndexedSubGraph> indexed_sub_graph,
                 PrivateTag)
    : OrtGraph(OrtGraphIrApi::kEpApi),
      graph_viewer_(*graph_viewer.get()),
      owned_graph_viewer_(std::move(graph_viewer)),
      owned_indexed_sub_graph_(std::move(indexed_sub_graph)) {}

// Static class function to create a std::unique_ptr<EpGraph>.
Status EpGraph::Create(const GraphViewer& graph_viewer, /*out*/ std::unique_ptr<EpGraph>& result) {
  auto ep_graph = std::make_unique<EpGraph>(graph_viewer, PrivateTag{});

  return CreateImpl(std::move(ep_graph), graph_viewer, result);
}

// Static class function to create a std::unique_ptr<EpGraph>.
Status EpGraph::Create(std::unique_ptr<GraphViewer> src_graph_viewer,
                       std::unique_ptr<IndexedSubGraph> src_indexed_sub_graph,
                       /*out*/ std::unique_ptr<EpGraph>& result) {
  auto& graph_viewer = *src_graph_viewer.get();
  auto ep_graph = std::make_unique<EpGraph>(std::move(src_graph_viewer),
                                            std::move(src_indexed_sub_graph),
                                            PrivateTag{});

  return CreateImpl(std::move(ep_graph), graph_viewer, result);
}

Status EpGraph::CreateImpl(std::unique_ptr<EpGraph> ep_graph, const GraphViewer& graph_viewer, /*out*/ std::unique_ptr<EpGraph>& result) {
  AllocatorPtr initializer_allocator = CPUAllocator::DefaultInstance();
  std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos_map;

  // Process graph inputs.
  const std::vector<const NodeArg*>& graph_input_node_args = graph_viewer.GetInputsIncludingInitializers();
  InlinedVector<EpValueInfo*> graph_input_value_infos(graph_input_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_input_node_args,
                              graph_input_value_infos,
                              [&graph_viewer](EpValueInfo* v) {
                                if (!graph_viewer.IsInitializedTensor(v->GetName())) {
                                  v->SetFlag(EpValueInfo::Flags::kIsRequiredGraphInput);
                                } else if (graph_viewer.CanOverrideInitializer()) {
                                  v->SetFlag(EpValueInfo::Flags::kIsOptionalGraphInput);
                                }
                              });

  // Process graph outputs.
  const std::vector<const NodeArg*>& graph_output_node_args = graph_viewer.GetOutputs();
  InlinedVector<EpValueInfo*> graph_output_value_infos(graph_output_node_args.size(), nullptr);
  ConvertNodeArgsToValueInfos(ep_graph.get(), value_infos_map, graph_output_node_args,
                              graph_output_value_infos,
                              [](EpValueInfo* v) { v->SetFlag(EpValueInfo::Flags::kIsGraphOutput); });

  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> outer_scope_initializer_values;

  // Create OrtValueInfo and OrtValue instances for each initializer.
  auto initializers_names = graph_viewer.GetAllInitializersNames();
  std::vector<EpValueInfo*> initializer_value_infos;
  std::unordered_map<std::string_view, std::unique_ptr<OrtValue>> initializer_values;

  initializer_value_infos.reserve(initializers_names.size());
  initializer_values.reserve(initializers_names.size());

  for (const auto& initializer_name : initializers_names) {
    EpValueInfo* value_info = nullptr;
    EpValueInfo::Flags flag = graph_viewer.IsConstantInitializer(initializer_name, /*check_outer_scope*/ false)
                                  ? EpValueInfo::kIsConstantInitializer
                                  : EpValueInfo::kIsOptionalGraphInput;

    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    graph_viewer.GetInitializedTensor(initializer_name, tensor_proto);

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

    initializer_value_infos.push_back(value_info);

    // Initialize OrtValue for the initializer.
    // Note: using std::unique_ptr<OrtValue> because we return a OrtValue* to the user and we want it to be stable.
    auto initializer_value = std::make_unique<OrtValue>();
    bool graph_has_ortvalue = graph_viewer.GetGraph().GetOrtValueInitializer(initializer_name, *initializer_value,
                                                                             /*check_outer_scope*/ false);

    if (!graph_has_ortvalue) {
      // Copy to OrtValue if not external. This should only happen for small initializers.
      // Do nothing for external initializers, as we will load/mmap into an OrtValue on demand.
      if (!utils::HasExternalDataInFile(*tensor_proto)) {
        ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), graph_viewer.ModelPath(), *tensor_proto,
                                                         initializer_allocator, *initializer_value));
      }
    }

    initializer_values.emplace(value_info->GetName(), std::move(initializer_value));
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

      // Note: using std::unique_ptr<OrtValue> because we return a OrtValue* to the user and we want it to be stable.
      auto outer_initializer_value = std::make_unique<OrtValue>();
      bool is_constant = false;
      const ONNX_NAMESPACE::TensorProto* outer_initializer = parent_graph->GetInitializer(implicit_name,
                                                                                          *outer_initializer_value,
                                                                                          is_constant,
                                                                                          /*check_outer_scope*/ true);
      outer_value_info->SetFlag(EpValueInfo::kIsOuterScope);

      if (outer_initializer != nullptr) {
        outer_value_info->SetFlag(is_constant ? EpValueInfo::kIsConstantInitializer : EpValueInfo::kIsOptionalGraphInput);
      }

      // Add the OrtValue if this is an initializer.
      if (outer_initializer != nullptr) {
        if (!outer_initializer_value->IsAllocated()) {
          // Copy to OrtValue if not external. This should only happen for small initializers.
          // Do nothing for external initializers. Will load/mmap into an OrtValue on demand.
          if (!utils::HasExternalDataInFile(*outer_initializer)) {
            ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), parent_graph->ModelPath(),
                                                             *outer_initializer, initializer_allocator,
                                                             *outer_initializer_value));
          }
        }
        outer_scope_initializer_values.emplace(outer_value_info->GetName(), std::move(outer_initializer_value));
      }
    }
  }

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

const ORTCHAR_T* EpGraph::GetModelPath() const {
  return graph_viewer_.ModelPath().c_str();
}

int64_t EpGraph::GetOnnxIRVersion() const { return graph_viewer_.GetOnnxIRVersion(); }

Status EpGraph::GetNumOperatorSets(size_t& num_operator_sets) const {
  num_operator_sets = graph_viewer_.DomainToVersionMap().size();
  return Status::OK();
}

Status EpGraph::GetOperatorSets(gsl::span<const char*> domains,
                                gsl::span<int64_t> opset_versions) const {
  const std::unordered_map<std::string, int>& domain_to_version = graph_viewer_.DomainToVersionMap();
  size_t num_operator_sets = domain_to_version.size();

  ORT_RETURN_IF_ERROR((CheckCopyDestination<const char*>("operator set domains", num_operator_sets, domains)));
  ORT_RETURN_IF_ERROR((CheckCopyDestination<int64_t>("operator set versions", num_operator_sets, opset_versions)));

  // Collect (domain, version) pairs and sort them by domain to ensure user always gets a stable ordering.
  std::vector<std::pair<const char*, int>> pairs;
  pairs.reserve(num_operator_sets);

  for (const auto& [domain, version] : domain_to_version) {
    pairs.emplace_back(domain.c_str(), version);
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<const char*, int>& a, const std::pair<const char*, int>& b) -> bool {
              return std::strcmp(a.first, b.first) < 0;
            });

  // Copy sorted (domain, version) pairs into the destination buffers.
  size_t index = 0;
  for (const auto& [domain_c_str, version] : pairs) {
    domains[index] = domain_c_str;
    opset_versions[index] = version;
    index++;
  }

  return Status::OK();
}

size_t EpGraph::GetNumInputs() const {
  return inputs_.size();
}

Status EpGraph::GetInputs(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_inputs = inputs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("graph inputs", num_inputs, dst)));

  for (size_t i = 0; i < num_inputs; ++i) {
    dst[i] = inputs_[i];
  }

  return Status::OK();
}

size_t EpGraph::GetNumOutputs() const {
  return outputs_.size();
}

Status EpGraph::GetOutputs(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_outputs = outputs_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("graph outputs", num_outputs, dst)));

  for (size_t i = 0; i < num_outputs; ++i) {
    dst[i] = outputs_[i];
  }

  return Status::OK();
}

size_t EpGraph::GetNumInitializers() const {
  return initializer_value_infos_.size();
}

Status EpGraph::GetInitializers(gsl::span<const OrtValueInfo*> dst) const {
  const size_t num_initializers = initializer_value_infos_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtValueInfo*>("graph initializers", num_initializers, dst)));

  for (size_t i = 0; i < num_initializers; ++i) {
    dst[i] = initializer_value_infos_[i];
  }

  return Status::OK();
}

size_t EpGraph::GetNumNodes() const {
  return nodes_.size();
}

Status EpGraph::GetNodes(gsl::span<const OrtNode*> dst) const {
  const size_t num_nodes = nodes_.size();
  ORT_RETURN_IF_ERROR((CheckCopyDestination<const OrtNode*>("graph nodes", num_nodes, dst)));

  for (size_t i = 0; i < num_nodes; ++i) {
    dst[i] = nodes_[i].get();
  }

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

Status EpGraph::GetInitializerValue(std::string_view name, const OrtValue*& result) const {
  auto ensure_ort_value_loaded = [&](const std::unique_ptr<OrtValue>& ort_value) -> Status {
    if (!ort_value->IsAllocated()) {
      // Lazy load the OrtValue. This happens for external initializers.
      const Graph& graph = graph_viewer_.GetGraph();
      ORT_RETURN_IF_ERROR(graph.LoadExternalInitializerAsOrtValue(std::string(name),
                                                                  const_cast<OrtValue&>(*ort_value)));
    }

    return Status::OK();
  };

  // Check for initializer value in the graph's scope.
  if (auto iter = initializer_values_.find(name);
      iter != initializer_values_.end()) {
    const std::unique_ptr<OrtValue>& ort_value = iter->second;
    ORT_RETURN_IF_ERROR(ensure_ort_value_loaded(ort_value));

    result = ort_value.get();
    return Status::OK();
  }

  // Check for the initializer value in an outer scope.
  // Only finds a value if the outer initializer value is used within this graph.
  if (auto iter = outer_scope_initializer_values_.find(name);
      iter != outer_scope_initializer_values_.end()) {
    const std::unique_ptr<OrtValue>& ort_value = iter->second;
    ORT_RETURN_IF_ERROR(ensure_ort_value_loaded(ort_value));

    result = ort_value.get();
    return Status::OK();
  }

  result = nullptr;
  return Status::OK();
}
}  // namespace onnxruntime
