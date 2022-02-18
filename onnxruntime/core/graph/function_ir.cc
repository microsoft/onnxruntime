#include <queue>

#include "core/graph/function_ir.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

using namespace ONNX_NAMESPACE;

#if !defined(ORT_MINIMAL_BUILD)
void FunctionIR::AddValueInfo(const NodeArg* new_value_info) {
  NodeArg* node_arg = GetNodeArg(new_value_info->Name());
  ORT_ENFORCE(node_arg && node_arg == new_value_info, "Error: trying to add an value info that are no in graph.");
  value_info_.insert(new_value_info);
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
Node& FunctionIR::AddNode(const Node& other) {
  const auto& definitions = other.GetDefinitions();

  auto& new_node = AddNode(other.Name(), other.OpType(), other.Description(),
                           definitions.input_defs,
                           definitions.output_defs,
                           &other.GetAttributes(),
                           other.Domain());

  return new_node;
}
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Node& FunctionIR::AddNode(const std::string& name,
    const std::string& op_type,
    const std::string& description,
    const std::vector<NodeArg*>& input_args,
    const std::vector<NodeArg*>& output_args,
    const NodeAttributes* attributes,
    const std::string& domain) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  inputs.resize(input_args.size());
  outputs.resize(output_args.size());
  int i = 0;
  for (auto input_arg : input_args) {
    inputs[i++] = &GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
  }
  i = 0;
  for (auto output_arg : output_args) {
    outputs[i++] = &GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
  }

  const gsl::not_null<Node*> node = AllocateNode();
  node->Init(name, op_type, description, inputs, outputs, attributes, domain);

  return *node;
}

bool FunctionIR::RemoveNode(NodeIndex node_index) {
  auto node = GetNode(node_index);
  if (nullptr == node) {
    return false;
  }

  // Node must be disconnected from any downstream nodes before removal
  ORT_ENFORCE(node->GetOutputEdgesCount() == 0, "Can't remove node ", node->Name(), " as it still has output edges.");

  // Remove all input edges.
  // Need to copy the edge info first so we can remove the real edges while iterating the copy of edge info.
  auto input_edges = node->GetRelationships().input_edges;

  for (auto& input_edge : input_edges) {
    RemoveEdge(input_edge.GetNode().Index(), node_index, input_edge.GetSrcArgIndex(), input_edge.GetDstArgIndex());
  }

  return ReleaseNode(node_index);
}

void FunctionIR::AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  if (nodes_.size() <= src_node_index || src_arg_slot < 0 || nodes_.size() <= dst_node_index || dst_arg_slot < 0 ||
      nullptr == nodes_[src_node_index] || nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ORT_THROW("Invalid node indexes specified when adding edge.");
  }

  NodeArg* src_arg = nullptr;
  NodeArg* dst_arg = nullptr;
  if (nodes_[src_node_index]->MutableDefinitions().output_defs.size() > static_cast<size_t>(src_arg_slot)) {
    src_arg = nodes_[src_node_index]->MutableDefinitions().output_defs[src_arg_slot];
  }

  if (nullptr == src_arg) {
    ORT_THROW("Invalid source node arg slot specified when adding edge.");
  }

  auto& dst_node_defs = nodes_[dst_node_index]->MutableDefinitions();
  NodeArg** dst_arg_pointer = nullptr;
  if (dst_node_defs.input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
    dst_arg_pointer = &dst_node_defs.input_defs[dst_arg_slot];
    dst_arg = *dst_arg_pointer;
  } else {
    auto num_of_explicit_inputs = dst_node_defs.input_defs.size();
    if (num_of_explicit_inputs + dst_node_defs.implicit_input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
      dst_arg_pointer = &dst_node_defs.implicit_input_defs[dst_arg_slot - num_of_explicit_inputs];
      dst_arg = *dst_arg_pointer;
    }
  }
  if (nullptr == dst_arg) {
    ORT_THROW("Invalid destination node arg slot specified when adding edge.");
  }

  if (src_arg != dst_arg) {
    if (src_arg->Type() != dst_arg->Type()) {
      // The output type of source node arg does not match the input type of destination node arg.
      ORT_THROW("Argument type mismatch when adding edge.");
    }
    *dst_arg_pointer = src_arg;
  }

  nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index], src_arg_slot, dst_arg_slot));
  nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index], src_arg_slot, dst_arg_slot));
}

void FunctionIR::RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  if (nodes_.size() <= src_node_index || src_arg_slot < 0 || nodes_.size() <= dst_node_index || dst_arg_slot < 0 ||
      nullptr == nodes_[src_node_index] || nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ORT_THROW("Invalid node indexes specified when removing edge.");
  }

  const NodeArg* src_arg = nullptr;
  const NodeArg* dst_arg = nullptr;
  if (nodes_[src_node_index]->GetDefinitions().output_defs.size() > static_cast<size_t>(src_arg_slot)) {
    src_arg = nodes_[src_node_index]->GetDefinitions().output_defs[src_arg_slot];
  }

  if (nullptr == src_arg) {
    ORT_THROW("Invalid source node arg slot specified when removing edge.");
  }

  auto& dst_node_defs = nodes_[dst_node_index]->GetDefinitions();
  if (dst_node_defs.input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
    dst_arg = dst_node_defs.input_defs[dst_arg_slot];
  } else {
    auto num_of_explicit_inputs = dst_node_defs.input_defs.size();
    if (num_of_explicit_inputs + dst_node_defs.implicit_input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
      dst_arg = dst_node_defs.implicit_input_defs[dst_arg_slot - num_of_explicit_inputs];
    }
  }
  if (nullptr == dst_arg) {
    ORT_THROW("Invalid destination node arg slot specified when removing edge.");
  }

  if (src_arg != dst_arg) {
    // The edge ends specified by source and destination arg slot are not referring to same node arg.
    // It means there was no edge between these two slots before.
    ORT_THROW("Argument mismatch when removing edge.");
  }

  nodes_[dst_node_index]->MutableRelationships().input_edges.erase(Node::EdgeEnd(*nodes_[src_node_index], src_arg_slot, dst_arg_slot));
  nodes_[src_node_index]->MutableRelationships().output_edges.erase(Node::EdgeEnd(*nodes_[dst_node_index], src_arg_slot, dst_arg_slot));
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
bool FunctionIR::AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index) {
  if (nodes_.size() <= src_node_index ||
      nodes_.size() <= dst_node_index ||
      nullptr == nodes_[src_node_index] ||
      nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    return false;
  }

  GSL_SUPPRESS(es .84) {  // ignoring return from insert()
    nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index]));
    nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index]));
    nodes_[dst_node_index]->MutableRelationships().control_inputs.insert(nodes_[src_node_index]->Name());
  }

  return true;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

void FunctionIR::ReverseDFSFrom(const std::vector<NodeIndex>& from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp) const {
  std::vector<const Node*> node_vec;
  node_vec.reserve(from.size());
  for (auto i : from) {
    node_vec.push_back(GetNode(i));
  }

  ReverseDFSFrom(node_vec, enter, leave, comp, {});
}

void FunctionIR::ReverseDFSFrom(const std::vector<const Node*>& from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp) const {
  ReverseDFSFrom(from, enter, leave, comp, {});
}

void FunctionIR::ReverseDFSFrom(const std::vector<const Node*>& from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp,
    const std::function<bool(const Node*, const Node*)>& stop) const {
  using WorkEntry = std::pair<const Node*, bool>;  // bool represents leave or not
  std::vector<WorkEntry> stack(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    stack[i] = WorkEntry(from[i], false);
  }

  std::vector<bool> visited(MaxNodeIndex(), false);
  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    if (last_entry.first == nullptr) {
      continue;
    }
    const Node& n = *last_entry.first;

    if (last_entry.second) {
      // leave node
      leave(&n);
      continue;
    }

    if (visited[n.Index()]) continue;

    visited[n.Index()] = true;

    if (enter) enter(&n);

    if (leave) stack.emplace_back(&n, true);

    if (comp) {
      std::vector<const Node*> sorted_nodes;
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        sorted_nodes.push_back(&(*iter));
      }
      std::sort(sorted_nodes.begin(), sorted_nodes.end(), comp);
      for (const auto* in : sorted_nodes) {
        const NodeIndex idx = in->Index();
        if (!visited[idx]) {
          stack.emplace_back(in, false);
        }
      }
    } else {
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        const NodeIndex idx = (*iter).Index();
        if (!visited[idx]) {
          stack.emplace_back(GetNode(idx), false);
        }
      }
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)
void FunctionIR::KahnsTopologicalSort(const std::function<void(const Node*)>& enter,
    const std::function<bool(const Node*, const Node*)>& comp) const {
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::priority_queue<const Node*, std::vector<const Node*>, decltype(comp)> to_visit(comp);
  std::vector<NodeIndex> topo_order;

  for (auto& node : Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    in_degree.insert({node.Index(), input_edge_count});
    if (input_edge_count == 0) {
      to_visit.push(&node);
    }
  }

  while (!to_visit.empty()) {
    const Node* current = to_visit.top();
    to_visit.pop();

    if (!current) continue;

    if (enter) {
      enter(current);
    }

    for (auto node_it = current->OutputNodesBegin(); node_it != current->OutputNodesEnd(); ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        to_visit.push(&*node_it);
      }
    }
    topo_order.push_back(current->Index());
  }

  if (NumberOfNodes() != static_cast<int>(topo_order.size())) {
    ORT_THROW("Some nodes are not included in the topological sort, graph have a cycle.");
  }
}

#endif

#if !defined(ORT_MINIMAL_BUILD)
void FunctionIR::SetInputs(const std::vector<const NodeArg*>& inputs) {
  graph_inputs_ = inputs;
}

void FunctionIR::SetOutputs(const std::vector<const NodeArg*>& outputs) {
  graph_outputs_ = outputs;
}

void FunctionIR::SetValueInfo(const std::unordered_set<const NodeArg*>& value_infos) {
  value_info_ = value_infos;
}
#endif


#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
/** Sets the type of a NodeArg, replacing existing type/shape if any */
void FunctionIR::SetNodeArgType(NodeArg& arg, const ONNX_NAMESPACE::TypeProto& type_proto) {
  arg.SetType(type_proto);
}
#endif

FunctionIR::~FunctionIR() {
}

Node& FunctionIR::AddNode(const ONNX_NAMESPACE::NodeProto& node_proto,
                          const ArgNameToTypeMap& name_to_type_map) {
  auto input_defs = CreateNodeArgs(node_proto.input(), name_to_type_map);
  auto output_defs = CreateNodeArgs(node_proto.output(), name_to_type_map);

  const int num_attributes = node_proto.attribute_size();
  NodeAttributes attributes;
  attributes.reserve(num_attributes);

  for (int i = 0; i < num_attributes; ++i) {
    auto& attr = node_proto.attribute(i);
    attributes[attr.name()] = attr;
  }

  return AddNode(node_proto.name(),
                 node_proto.op_type(),
                 node_proto.doc_string(),
                 input_defs,
                 output_defs,
                 &attributes,
                 node_proto.domain());
}

#if !defined(ORT_MINIMAL_BUILD)
std::vector<NodeArg*> FunctionIR::CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
    const ArgNameToTypeMap& name_to_type_map) {
  const auto name_to_type_map_end = name_to_type_map.end();
  std::vector<NodeArg*> results;
  results.reserve(names.size());

  for (auto& name : names) {
    const TypeProto* type = nullptr;

    auto name_to_type_iter = name_to_type_map.find(name);
    if (name_to_type_iter != name_to_type_map_end) {
      // This node input arg type/shape does exist in graph proto.
      // Assign type/shape information to node input arg.
      type = &(name_to_type_iter->second);
    }

    auto node_arg = &GetOrCreateNodeArg(name, type);
    results.push_back(node_arg);
  }

  return results;
}
#endif


#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
Status FunctionIR::PopulateNodeArgToProducerConsumerLookupsFromNodes() {
  node_arg_to_producer_node_.clear();
  node_arg_to_consumer_nodes_.clear();

  for (const auto& node : Nodes()) {
    node.ForEachDef([&](const NodeArg& node_arg, bool is_input) {
      if (is_input) {
        node_arg_to_consumer_nodes_[node_arg.Name()].insert(node.Index());
      } else {
        node_arg_to_producer_node_.insert({node_arg.Name(), node.Index()});
      }
    });
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// calling private ctor
GSL_SUPPRESS(r .11)
gsl::not_null<Node*> FunctionIR::AllocateNode() {
  ORT_ENFORCE(nodes_.size() < static_cast<unsigned int>(std::numeric_limits<int>::max()));
  std::unique_ptr<Node> new_node(new Node(nodes_.size(), *graph_));
  Node* node{new_node.get()};

  nodes_.push_back(std::move(new_node));
  ++num_of_nodes_;

  return gsl::not_null<Node*>{node};
}

// TODO: Does this need (and maybe AllocateNode) to be threadsafe so nodes_ and num_of_nodes_ managed more carefully?
bool FunctionIR::ReleaseNode(NodeIndex index) {
  if (index >= nodes_.size()) {
    return false;
  }

  // index is valid, but the entry may already be empty
  if (nodes_[index] != nullptr) {
    nodes_[index] = nullptr;
    --num_of_nodes_;
  }

  return true;
}

Status FunctionIR::MoveToTarget(const IndexedSubGraph& sub_graph, FunctionIR* target) {
  //1. move the nodes into target function
  //2. move the node args that owned by target function
  //3. update the consumer / producer
}
#endif
}