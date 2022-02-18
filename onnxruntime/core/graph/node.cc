// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/graph_flatbuffers_utils.h"

namespace onnxruntime {

using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ::onnxruntime::common;

Node::EdgeEnd::EdgeEnd(const Node& node, int src_arg_index, int dst_arg_index) noexcept
    : node_(&node),
      src_arg_index_(src_arg_index),
      dst_arg_index_(dst_arg_index) {
}

Node::EdgeEnd::EdgeEnd(const Node& node) noexcept
    : EdgeEnd(node, INT_MAX, INT_MAX) {
}

Node::NodeConstIterator::NodeConstIterator(EdgeConstIterator p_iter) {
  m_iter = p_iter;
}

bool Node::NodeConstIterator::operator==(const NodeConstIterator& p_other) const {
  return m_iter == p_other.m_iter;
}

bool Node::NodeConstIterator::operator!=(const NodeConstIterator& p_other) const {
  return m_iter != p_other.m_iter;
}

void Node::NodeConstIterator::operator++() {
  ++m_iter;
}

void Node::NodeConstIterator::operator--() {
  --m_iter;
}

const Node& Node::NodeConstIterator::operator*() const {
  return (*m_iter).GetNode();
}

const Node* Node::NodeConstIterator::operator->() const {
  return &(operator*());
}

void Node::SetPriority(int priority) noexcept {
  priority_ = priority;
}

//TODO!!!
//Why node need to have path?
//const Path& Node::ModelPath() const noexcept {
//  return graph_->ModelPath();
//}

#if !defined(ORT_MINIMAL_BUILD)
//TODO!!!
//Re-implement Function
//Function* Node::GetMutableFunctionBody(bool try_init_func_body) {
//  if (nullptr != func_body_) {
//    return func_body_;
//  }
//
//  // Initialize function body
//  if (try_init_func_body) {
//    graph_->InitFunctionBodyForNode(*this);
//  }
//
//  return func_body_;
//}
//
//void Node::SetFunctionBody(Function& func) {
//  func_body_ = &func;
//  op_ = &func.OpSchema();
//  since_version_ = op_->since_version();
//}

//TODO: move it to GraphProtoSerializer
void Node::ToProto(NodeProto& proto, bool update_subgraphs) const {
  proto.set_name(name_);
  proto.set_op_type(op_type_);

  if (!domain_.empty())
    proto.set_domain(domain_);

  if (!description_.empty())
    proto.set_doc_string(description_);

  // Set attributes.
  proto.clear_attribute();
  for (const auto& attribute : attributes_) {
    const gsl::not_null<AttributeProto*> attr{proto.add_attribute()};
    *attr = attribute.second;  // copy
    if (update_subgraphs && attr->has_g()) {
      attr->clear_g();
      *attr->mutable_g() = attr_to_subgraph_map_.find(attribute.first)->second->ToGraphProto();
    }
  }

  // Set inputs' definitions.
  proto.clear_input();
  for (auto& input_def : definitions_.input_defs) {
    proto.add_input(input_def->Name());
  }

  // Set outputs' definitions.
  proto.clear_output();
  for (auto& output_def : definitions_.output_defs) {
    proto.add_output(output_def->Name());
  }
}

//TODO!!!!
//Move it to ORTFormatter
Status Node::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                             flatbuffers::Offset<onnxruntime::fbs::Node>& fbs_node) const {
  // if type is Primitive it's an ONNX function and currently we have kernel implementations for all those
  // TODO!!!
  // Re-implement it
  /*if (func_body_ != nullptr && node_type_ != Type::Primitive) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Serialization of fused function body is not currently supported, ",
                           "Node [", name_, "] op_type [", op_type_, "]");
  }*/

  auto GetNodeArgsOrtFormat = [&builder](const std::vector<NodeArg*>& src) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> node_args(src.size());
    std::transform(src.cbegin(), src.cend(), node_args.begin(),
                   [&builder](const NodeArg* nodearg) {
                     // NodeArg's name will be used by multiple places, create shared string
                     return builder.CreateSharedString(nodearg->Name());
                   });
    return builder.CreateVector(node_args);
  };

  auto name = builder.CreateString(name_);
  auto doc_string = builder.CreateString(description_);
  auto domain = builder.CreateSharedString(domain_);
  auto op_type = builder.CreateSharedString(op_type_);
  auto ep = builder.CreateSharedString(execution_provider_type_);
  auto inputs = GetNodeArgsOrtFormat(definitions_.input_defs);
  auto outputs = GetNodeArgsOrtFormat(definitions_.output_defs);
  auto input_arg_counts = builder.CreateVector(definitions_.input_arg_count);
  auto implicit_inputs = GetNodeArgsOrtFormat(definitions_.implicit_input_defs);

  // Node attributes
  std::vector<flatbuffers::Offset<fbs::Attribute>> attributes_vec;
  attributes_vec.reserve(attributes_.size());
  for (const auto& entry : attributes_) {
    const auto& attr_name = entry.first;
    const auto& attr_proto = entry.second;
    flatbuffers::Offset<fbs::Attribute> fbs_attr;
    Graph* subgraph = nullptr;
    if (attr_proto.has_g()) {
      const auto it = attr_to_subgraph_map_.find(attr_name);
      ORT_RETURN_IF_NOT(it != attr_to_subgraph_map_.cend(),
                        "Node [", name_, "] op_type [", op_type_, "] ", "does not have the graph for key ", attr_name);
      subgraph = it->second;
    }
    //TODO!!!
    //Fix the model path later
    ORT_RETURN_IF_ERROR(
        fbs::utils::SaveAttributeOrtFormat(builder, attr_proto, fbs_attr, Path(), subgraph));
    attributes_vec.push_back(fbs_attr);
  }
  auto attributes = builder.CreateVector(attributes_vec);

  fbs::NodeBuilder nb(builder);
  nb.add_name(name);
  nb.add_doc_string(doc_string);
  nb.add_domain(domain);
  nb.add_since_version(since_version_);
  nb.add_index(gsl::narrow<uint32_t>(index_));
  nb.add_op_type(op_type);
  nb.add_type(static_cast<fbs::NodeType>(node_type_));
  nb.add_execution_provider_type(ep);
  nb.add_inputs(inputs);
  nb.add_outputs(outputs);
  nb.add_attributes(attributes);
  nb.add_input_arg_counts(input_arg_counts);
  nb.add_implicit_inputs(implicit_inputs);
  fbs_node = nb.Finish();
  return Status::OK();
}

flatbuffers::Offset<fbs::NodeEdge> Node::SaveEdgesToOrtFormat(flatbuffers::FlatBufferBuilder& builder) const {
  auto get_edges = [](const EdgeSet& edge_set) {
    std::vector<fbs::EdgeEnd> edges;
    edges.reserve(edge_set.size());
    for (const auto& edge : edge_set)
      edges.push_back(fbs::EdgeEnd(gsl::narrow<uint32_t>(edge.GetNode().Index()),
                                   edge.GetSrcArgIndex(), edge.GetDstArgIndex()));

    return edges;
  };

  const auto input_edges = get_edges(relationships_.input_edges);
  const auto output_edges = get_edges(relationships_.output_edges);
  return fbs::CreateNodeEdgeDirect(builder, gsl::narrow<uint32_t>(index_), &input_edges, &output_edges);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

//TODO!!!
//Move it to ORTFormatter
Status Node::LoadFromOrtFormat(const onnxruntime::fbs::Node& fbs_node, Graph& graph,
                               const logging::Logger& logger, std::unique_ptr<Node>& node) {
  node = std::make_unique<Node>(fbs_node.index(), graph);
  return node->LoadFromOrtFormat(fbs_node, logger);
}

Status Node::LoadFromOrtFormat(const onnxruntime::fbs::Node& fbs_node, const logging::Logger& logger) {
  auto LoadNodeArgsFromOrtFormat =
      [&](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* fbs_node_arg_names,
          std::vector<NodeArg*>& node_args,
          bool check_parent_graph = false) -> Status {
    ORT_RETURN_IF(nullptr == fbs_node_arg_names, "fbs_node_arg_names cannot be null");
    node_args.reserve(fbs_node_arg_names->size());
    for (const auto* node_arg_name : *fbs_node_arg_names) {
      ORT_RETURN_IF(nullptr == node_arg_name, "node_arg_name cannot be null");
      auto* node_arg = check_parent_graph ? graph_->GetNodeArgIncludingParentGraphs(node_arg_name->str())
                                          : graph_->GetNodeArg(node_arg_name->str());
      ORT_RETURN_IF(nullptr == node_arg, "LoadNodeArgsFromOrtFormat: Node [", name_, "] op_type [", op_type_,
                    "], could not find NodeArg ", node_arg_name->str());
      node_args.push_back(node_arg);
    }
    return Status::OK();
  };

  // index_ was set in the ctor of this Node instance
  fbs::utils::LoadStringFromOrtFormat(name_, fbs_node.name());
  fbs::utils::LoadStringFromOrtFormat(description_, fbs_node.doc_string());
  fbs::utils::LoadStringFromOrtFormat(domain_, fbs_node.domain());
  since_version_ = fbs_node.since_version();
  fbs::utils::LoadStringFromOrtFormat(op_type_, fbs_node.op_type());
  node_type_ = static_cast<Node::Type>(fbs_node.type());
  // we skip populating the saved EP here
  // the node will either be assigned to another EP by the ORT format model-specific graph partitioning or fall back to
  // the EP encoded in its kernel def hash
  // fbs::utils::LoadStringFromOrtFormat(execution_provider_type_, fbs_node.execution_provider_type());
  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.inputs(), definitions_.input_defs));

  // attributes
  auto fbs_attributes = fbs_node.attributes();
  if (fbs_attributes) {
    for (const auto* fbs_attr : *fbs_attributes) {
      ORT_RETURN_IF(nullptr == fbs_attr, "fbs_attr cannot be null");
      AttributeProto attr_proto;
      std::unique_ptr<Graph> subgraph;
      ORT_RETURN_IF_ERROR(
          fbs::utils::LoadAttributeOrtFormat(*fbs_attr, attr_proto, subgraph, *graph_, *this, logger));

      // If we have a sub graph in this attributes, it will be loaded into subgraph ptr
      // while the attribute proto contains the sub graph will have the empty g() field
      if (attr_proto.type() == AttributeProto_AttributeType_GRAPH) {
        ORT_RETURN_IF_NOT(subgraph, "Serialization error. Graph attribute was serialized without Graph instance");
        attr_to_subgraph_map_.emplace(attr_proto.name(), gsl::not_null<Graph*>(subgraph.get()));
        subgraphs_.push_back(std::move(subgraph));
      }

      AddAttribute(attr_proto.name(), attr_proto);
    }
  }

  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.implicit_inputs(), definitions_.implicit_input_defs,
                                                /* check parent graphs */ true));

  {  // input_arg_counts
    auto fbs_input_arg_counts = fbs_node.input_arg_counts();
    ORT_RETURN_IF(nullptr == fbs_input_arg_counts, "Node::LoadFromOrtFormat, input_arg_counts is missing");
    auto& input_arg_count = definitions_.input_arg_count;
    input_arg_count.reserve(fbs_input_arg_counts->size());
    input_arg_count.insert(input_arg_count.begin(), fbs_input_arg_counts->cbegin(), fbs_input_arg_counts->cend());
  }

  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.outputs(), definitions_.output_defs));

  return Status::OK();
}

Status Node::LoadEdgesFromOrtFormat(const onnxruntime::fbs::NodeEdge& fbs_node_edges,
                                    const Graph& graph) {
  ORT_RETURN_IF(fbs_node_edges.node_index() != index_,
                "input index: ", fbs_node_edges.node_index(), " is not the same as this node's index:", index_);

  auto add_edges = [&graph](const flatbuffers::Vector<const onnxruntime::fbs::EdgeEnd*>* fbs_edges,
                            EdgeSet& edge_set, const std::string& dst_name) -> Status {
    if (fbs_edges) {
      for (const auto* fbs_edge : *fbs_edges) {
        ORT_RETURN_IF(nullptr == fbs_edge, "Node::LoadEdgesFromOrtFormat, edge is missing for ", dst_name);
        edge_set.emplace(*graph.GetNode(fbs_edge->node_index()), fbs_edge->src_arg_index(), fbs_edge->dst_arg_index());
      }
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(add_edges(fbs_node_edges.input_edges(), relationships_.input_edges, "input edges"));
  ORT_RETURN_IF_ERROR(add_edges(fbs_node_edges.output_edges(), relationships_.output_edges, "output edges"));

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void Node::Init(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const NodeAttributes* attributes,
                const std::string& domain) {
  name_ = name;
  op_type_ = op_type;
  description_ = description;
  definitions_.input_defs = input_args;
  definitions_.output_defs = output_args;
  domain_ = domain;
  priority_ = 0;
  if (kOnnxDomainAlias == domain_) {
    domain_ = kOnnxDomain;
  }

  // Set each arg count as 1 by default.
  // It could be adjusted when resolving the node with its operator
  // information.
  definitions_.input_arg_count.assign(input_args.size(), 1);

  if (attributes) {
    attributes_ = *attributes;

    for (auto& name_to_attr : attributes_) {
      if (utils::HasGraph(name_to_attr.second)) {
#if !defined(ORT_MINIMAL_BUILD)
        CreateSubgraph(name_to_attr.first);
#else
        ORT_THROW("Creating node with a subgraph via AddNode is not supported in this build.");
#endif
      }
    }
  }
}

Node::Definitions& Node::MutableDefinitions() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return definitions_;
}

Node::Relationships& Node::MutableRelationships() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return relationships_;
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
void Node::CreateSubgraph(const std::string& attr_name) {
  auto attr = attributes_.find(attr_name);

  if (attr != attributes_.cend() && utils::HasGraph(attr->second)) {
    GraphProto& mutable_graph = *attr->second.mutable_g();
    std::unique_ptr<Graph> subgraph = std::make_unique<Graph>(*graph_, *this, mutable_graph);
    attr_to_subgraph_map_.insert({std::string(attr_name), gsl::not_null<Graph*>{subgraph.get()}});
    subgraphs_.emplace_back(std::move(subgraph));
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

void Node::AddAttribute(const std::string& attr_name, const AttributeProto& value) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  attributes_[attr_name] = value;
}

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                           \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    a.set_##field(value);                                                    \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_ATTR_IMPL(type, enumType, field)                                 \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    *(a.mutable_##field()) = value;                                          \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_LIST_ATTR_IMPL(type, enumType, field)            \
  void Node::AddAttribute(const std::string& attr_name,      \
                          const std::vector<type>& values) { \
    graph_->SetGraphResolveNeeded();                         \
    graph_->SetGraphProtoSyncNeeded();                       \
    AttributeProto a;                                        \
    a.set_name(attr_name);                                   \
    a.set_type(enumType);                                    \
    for (const auto& val : values) {                         \
      *(a.mutable_##field()->Add()) = val;                   \
    }                                                        \
    attributes_[attr_name] = a;                              \
  };

void Node::AddAttribute(const std::string& attr_name, const GraphProto& value) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  AttributeProto a;
  a.set_name(attr_name);
  a.set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH);
  *a.mutable_g() = value;
  attributes_[attr_name] = a;

#if !defined(ORT_MINIMAL_BUILD)
  // subgraph is created via deserialization and not here in a minimal build
  CreateSubgraph(attr_name);
#endif
};

ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR, t)
ADD_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTO, tp)
ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)
ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS, graphs)
ADD_LIST_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTOS, type_protos)
#if !defined(DISABLE_SPARSE_TENSORS)
ADD_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSOR, sparse_tensor)
ADD_LIST_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSORS, sparse_tensors)
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
bool Node::ClearAttribute(const std::string& attr_name) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return attributes_.erase(attr_name) > 0;
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
Status Node::UpdateInputArgCount() {
  // The node refers to a primitive operator.
  // Infer and verify node input arg type information.
  int total_arg_count = std::accumulate(definitions_.input_arg_count.cbegin(),
                                        definitions_.input_arg_count.cend(), 0);

  if (total_arg_count < 0 || static_cast<size_t>(total_arg_count) != definitions_.input_defs.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "This is an invalid model. "
                           "The sum of input arg count is not equal to size of input defs in node (",
                           name_, ")");
  }

  // op_ is always valid when this is called
  const ONNX_NAMESPACE::OpSchema& op = *Op();

  // Verify size of node arg count is same as input number in
  // operator definition.
  if (op.inputs().size() != definitions_.input_arg_count.size()) {
    // Adjust input arg count array with op definition
    // The adjustment will work as below,
    // In total, there're <total_arg_count> inputs, which
    // will be split as <1, 1, 1, 1, ... 1, x> or
    // <1, 1, 1, 1, ...1, 0, 0, ...0>. The final input
    // arg count array's element number will be the same
    // as op definition, and the sum of all elements will
    // be equal to <total_arg_count>.
    auto& input_arg_count = definitions_.input_arg_count;
    input_arg_count.clear();
    size_t m = 0;
    auto arg_count_left = total_arg_count;

    if (!op.inputs().empty()) {
      for (; m < op.inputs().size() - 1; ++m) {
        if (arg_count_left > 0) {
          input_arg_count.push_back(1);
          arg_count_left--;
        } else {
          input_arg_count.push_back(0);
        }
      }
    }

    // Set the arg count for the last input formal parameter.
    // NOTE: in the case that there's no .input(...) defined
    // in op schema, all input args will be fed as one input
    // of the operator.
    input_arg_count.push_back(arg_count_left);

    graph_->SetGraphResolveNeeded();
    graph_->SetGraphProtoSyncNeeded();
  }

  return Status::OK();
}

Graph* Node::GetMutableGraphAttribute(const std::string& attr_name) {
  Graph* subgraph = nullptr;

  const auto& entry = attr_to_subgraph_map_.find(attr_name);
  if (entry != attr_to_subgraph_map_.cend()) {
    subgraph = entry->second;
  }

  return subgraph;
}

const Graph* Node::GetGraphAttribute(const std::string& attr_name) const {
  return const_cast<Node*>(this)->GetMutableGraphAttribute(attr_name);
}

void Node::ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements) {
  std::vector<std::vector<NodeArg*>*> all_defs = {&definitions_.input_defs, &definitions_.output_defs};

  for (auto pair : replacements)
    for (auto* defs : all_defs)
      for (auto& def : *defs)
        if (def == pair.first)
          def = pair.second;
}

#endif  // !defined(ORT_MINIMAL_BUILD)

std::vector<gsl::not_null<const Graph*>> Node::GetSubgraphs() const {
  std::vector<gsl::not_null<const Graph*>> subgraphs;
  subgraphs.reserve(attr_to_subgraph_map_.size());
  using value_type = std::unordered_map<std::string, gsl::not_null<Graph*>>::value_type;
  std::transform(attr_to_subgraph_map_.cbegin(), attr_to_subgraph_map_.cend(), std::back_inserter(subgraphs),
                 [](const value_type& entry) { return entry.second; });

  return subgraphs;
}

std::unordered_map<std::string, gsl::not_null<const Graph*>> Node::GetAttributeNameToSubgraphMap() const {
  std::unordered_map<std::string, gsl::not_null<const Graph*>> attr_to_subgraphs;
  for (auto& entry : attr_to_subgraph_map_) {
    attr_to_subgraphs.insert({entry.first, entry.second});
  }
  return attr_to_subgraphs;
}

void Node::ForEachDef(std::function<void(const onnxruntime::NodeArg&, bool is_input)> func,
                      bool include_missing_optional_defs) const {
  for (const auto* arg : InputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, true);
  }

  for (const auto* arg : ImplicitInputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, true);
  }

  for (const auto* arg : OutputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, false);
  }
};

}  // namespace onnxruntime
