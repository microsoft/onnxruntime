#include "core/graph/graph_context.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

using namespace ONNX_NAMESPACE;
using google::protobuf::RepeatedPtrField;

constexpr char* kMainFuncKey = "@MAIN_FUNC@";

template <typename T, typename TIter>
static void RemoveRepeatedFieldEntry(T& repeated_field, const TIter& entry_to_remove) {
  auto num_entries = repeated_field.size();
  if (num_entries > 1) {
    // swap the entry being deleted with the last one, and delete it.
    // we do this so we don't have to move all the entries past the one being deleted down one.
    auto slot = entry_to_remove - repeated_field.begin();
    auto last_entry = repeated_field.end() - 1;
    repeated_field.SwapElements(gsl::narrow<int>(slot), gsl::narrow<int>(num_entries - 1));
    repeated_field.erase(last_entry);
  } else {
    repeated_field.erase(entry_to_remove);
  }
}

static TypeProto TypeProtoFromTensorProto(const TensorProto& tensor) {
  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(tensor.data_type());
  auto shape = t.mutable_tensor_type()->mutable_shape();
  for (auto dim : tensor.dims())
    shape->add_dim()->set_dim_value(dim);

  return t;
}

static std::string GetMapKey(const std::string& name, const std::string& domain) {
  std::string key(name);
  key.append(1, '_').append(domain.empty() ? kOnnxDomainAlias : domain);
  return key;
}

const FunctionIR& GraphContext::GetFunction(const std::string& name, const std::string& domain) const {
  auto it = name_to_functions_.find(GetMapKey(name, domain));
  if (it == name_to_functions_.end())
    ORT_THROW("Target function not found in the graph. Function name: ", name, ", domain: ", domain);
  return *(it->second);
}

FunctionIR* GraphContext::GetMutableFunction(const std::string& name, const std::string& domain) {
  auto it = name_to_functions_.find(GetMapKey(name, domain));
  if (it == name_to_functions_.end())
    ORT_THROW("Target function not found in the graph. Function name: ", name, ", domain: ", domain);
  return it->second.get();
}

const FunctionIR& GraphContext::GetMainFunction() const {
  auto it = name_to_functions_.find(kMainFuncKey);
  if (it == name_to_functions_.end())
    ORT_THROW("Main function not found in the graph.");
  return *it->second;
}

FunctionIR* GraphContext::GetMutableMainFunction() {
  auto it = name_to_functions_.find(kMainFuncKey);
  if (it == name_to_functions_.end())
    ORT_THROW("Main function not found in the graph.");
  return it->second.get();
}

Status GraphContext::AddFunction(const std::string& name, const std::string& domain, std::unique_ptr<FunctionIR> p_function) {
  auto key = GetMapKey(name, domain);
  auto it = name_to_functions_.find(key);
  if (it != name_to_functions_.end())
    return Status(common::ONNXRUNTIME, common::FAIL, 
                  "Function already exist in the graph context, \
                   please give an unique name/domain.           \
                   Function name: " + name + ",                  \
                   domain: " + domain);
  name_to_functions_.insert({key, std::move(p_function)});
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
Status GraphContext::ReplaceInitializedTensor(const ONNX_NAMESPACE::TensorProto& new_initializer) {
  // name_to_initial_tensor_ maps from name to const TensorProto*, so we first
  // look up the const pointer by name, then find and modify the mutable
  // pointed-to TensorProto in graph_proto_.
  const auto& initializer_name = new_initializer.name();
  const auto name_to_initializer_it = name_to_initial_tensor_.find(initializer_name);
  ORT_RETURN_IF_NOT(name_to_initializer_it != name_to_initial_tensor_.end(),
                    "Failed to find existing initializer with name ", initializer_name, ".");

  const auto& old_initializer = *(name_to_initializer_it->second);

  auto dims_eq = [&old_initializer, &new_initializer]() {
    if (old_initializer.dims_size() != new_initializer.dims_size()) return false;
    for (int i = 0; i < old_initializer.dims_size(); ++i) {
      if (old_initializer.dims(i) != new_initializer.dims(i)) return false;
    }
    return true;
  };

  ORT_RETURN_IF_NOT(dims_eq(), "Replacement tensor's dimensions do not match.");
  ORT_RETURN_IF_NOT(old_initializer.data_type() == new_initializer.data_type(),
                    "Replacement tensor's data type does not match.");

  auto& mutable_initializers = *(graph_proto_->mutable_initializer());
  // use cheaper pointer comparison to find old entry
  auto existing_entry = std::find(mutable_initializers.pointer_begin(), mutable_initializers.pointer_end(),
                                  &old_initializer);

  // these should always be in sync as the pointer in name_to_initial_tensor_ is to memory owned by graph_proto_
  ORT_ENFORCE(existing_entry != mutable_initializers.pointer_end(),
              "graph_proto_ is not in sync with name_to_initial_tensor_");

  **existing_entry = new_initializer;

  return Status::OK();
}
#endif

void GraphContext::RemoveInitializedTensor(const std::string& tensor_name) {
  bool found = false;
  auto iter = name_to_initial_tensor_.find(tensor_name);
  found = iter != name_to_initial_tensor_.end();
  if (found) {
    name_to_initial_tensor_.erase(iter);
#if !defined(DISABLE_SPARSE_TENSORS)
    sparse_tensor_names_.erase(tensor_name);
#endif
  } else {
#if !defined(DISABLE_SPARSE_TENSORS)
    ORT_ENFORCE(sparse_tensor_names_.count(tensor_name) == 0, "sparse_tensor_names_ not in sync with name_to_initial_tensor_");
#endif
  }

  auto& mutable_initializers = *(graph_proto_->mutable_initializer());
  auto proto_entry = std::find_if(mutable_initializers.begin(), mutable_initializers.end(),
                                  [&tensor_name](const TensorProto& entry) { return entry.name() == tensor_name; });

  if (proto_entry != mutable_initializers.end()) {
    RemoveRepeatedFieldEntry(mutable_initializers, proto_entry);
  } else {
    // these should always be in sync as the pointer in name_to_initial_tensor_ is to memory owned by graph_proto_
    ORT_ENFORCE(!found, "graph_proto_ is not in sync with name_to_initial_tensor_.");
  }
}

void GraphContext::CleanAllInitializedTensors() noexcept {
  name_to_initial_tensor_.clear();
#if !defined(DISABLE_SPARSE_TENSORS)
  sparse_tensor_names_.clear();
#endif

  // Clearing RepeatedPtrFields does not free objects' memory. The memory is retained
  // and can be reused. Need to explicitly release the cleared objects and free the
  // memory.
  graph_proto_->mutable_initializer()->Clear();
  const int num_cleared = graph_proto_->initializer().ClearedCount();
  for (int i = 0; i < num_cleared; i++) {
    delete graph_proto_->mutable_initializer()->ReleaseCleared();
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void GraphContext::AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor) {
  auto existing = name_to_initial_tensor_.find(tensor.name());
  if (existing != name_to_initial_tensor_.cend()) {
    ORT_ENFORCE(existing->second == &tensor,
                "AddInitializedTensor already has tensor with name ", tensor.name(), " but different TensorProto.");
    return;
  }

  const gsl::not_null<TensorProto*> tensor_added{graph_proto_->add_initializer()};
  *(tensor_added) = tensor;
  name_to_initial_tensor_[tensor.name()] = tensor_added;
}
#endif

const ONNX_NAMESPACE::TensorProto* GraphContext::GetInitializer(const std::string& name) const {
  auto iter = name_to_initial_tensor_.find(name);
  if (name_to_initial_tensor_.end() == iter) {
    return nullptr;
  } else {
    return iter->second;
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
bool GraphContext::IsSparseInitializer(const std::string& name) const {
  return sparse_tensor_names_.count(name) > 0;
}
#endif

GraphContext::GraphContext(ONNX_NAMESPACE::GraphProto* graph_proto, 
    const Path& model_path, 
    Graph* graph, 
    Version ir_version,
    bool is_subgraph,
    const logging::Logger& logger) : graph_proto_(graph_proto) {
  // create main function
  name_to_functions_.insert({kMainFuncKey, std::make_unique<FunctionIR>(graph)});
  // Process 'Constant' nodes
  // Put the 'TensorProto' stored in the 'Constant' nodes attribute into the graphs initializer list
  for (auto& node : graph_proto_->node()) {
    if (node.op_type() != kConstant) {
      continue;
    }

    const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
    auto status = utils::ConstantNodeProtoToTensorProto(node, model_path, *tensor);
    ORT_ENFORCE(status.IsOK(), status.ToString());
    // Ensure initializers are also graph inputs.
    if (ir_version < 4) {
      TypeProto t{TypeProtoFromTensorProto(*tensor)};
      const NodeArg& node_arg = GetMutableMainFunction()->GetOrCreateNodeArg(tensor->name(), &t);
      *(graph_proto->add_input()) = node_arg.ToProto();
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    if (node.attribute(0).type() == AttributeProto_AttributeType_SPARSE_TENSOR) {
      auto p = sparse_tensor_names_.emplace(tensor->name());
      ORT_ENFORCE(p.second, "Duplicate constant node sparse initializer name: '", tensor->name(), "' Model is invalid.");
    }
#endif
  }

  // Remove constant nodes as they're replaced with initializers above.
  const gsl::not_null<RepeatedPtrField<NodeProto>*> graph_mutable_nodes{graph_proto_->mutable_node()};
  graph_mutable_nodes->erase(
      std::remove_if(graph_mutable_nodes->begin(), graph_mutable_nodes->end(),
                     [](NodeProto& p) {
                       return (p.op_type() == kConstant);
                     }),
      graph_mutable_nodes->end());

#if !defined(DISABLE_SPARSE_TENSORS)
  // For now we convert sparse_intializer to dense tensors
  // since there are currently no supported ops that consume sparse
  // initializers directly. We remove them from graph_proto. We will reconstitute them
  // when saving to ORT format to save space on disk.
  if (graph_proto_->sparse_initializer_size() > 0) {
    for (const auto& sparse_tensor : graph_proto_->sparse_initializer()) {
      ORT_ENFORCE(utils::HasName(sparse_tensor), "Sparse initializer must have a name. This model is invalid");
      const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
      auto status = utils::SparseTensorProtoToDenseTensorProto(sparse_tensor, model_path, *tensor);
      ORT_ENFORCE(status.IsOK(), status.ToString());
      auto p = sparse_tensor_names_.emplace(tensor->name());
      ORT_ENFORCE(p.second, "Duplicate sparse_tensor_initializer: '", tensor->name(), "' Model is invalid.");
    }

    // Remove sparse_initializers from protobuf to save memory as they are converted to dense now
    graph_proto_->mutable_sparse_initializer()->Clear();
    const int sparse_num_cleared = graph_proto_->sparse_initializer().ClearedCount();
    for (int i = 0; i < sparse_num_cleared; ++i) {
      delete graph_proto_->mutable_sparse_initializer()->ReleaseCleared();
    }
  }
#endif
  ArgNameToTypeMap name_to_type_map;

  //TODO!!!
  //Re-implement function
  /*for (auto func : model_functions) {
    model_local_functions_[function_utils::GetFunctionIdentifier(func->domain(), func->name())] = func;
  }*/

  // Collect all node arg name, type, shape information in the graph.
  // type/shape information will be assigned to each node arg when going
  // thru all nodes later.

  // process graph inputs first as we want the type/shape from them to be preferred if a graph input
  // has a matching initializer
  for (auto& graph_input : graph_proto_->input()) {
    if (utils::HasName(graph_input)) {
      if (utils::HasType(graph_input)) {
        name_to_type_map[graph_input.name()] = graph_input.type();
        GetMutableMainFunction()->GetOrCreateNodeArg(graph_input.name(), &graph_input.type());
      } else {
        // subgraph inputs can have type inferred later. need to create a NodeArg in case this input is only used in
        // a nested subgraph (a NodeArg won't be added by AddNode for the nodes in this subgraph)
        if (is_subgraph) {
          GetMutableMainFunction()->GetOrCreateNodeArg(graph_input.name(), nullptr);
        }
      }
    }
  }

  // Copy initial tensors to a map.
  for (auto& tensor : graph_proto_->initializer()) {
    auto p = name_to_initial_tensor_.emplace(tensor.name(), &tensor);
    if (!p.second) {
      LOGS(logger, WARNING) << "Duplicate initializer (dense, sparse or ConstantNode): '" << tensor.name()
                             << "' the model will use the latest encountered initializer"
                             << ". Please, fix your model.";
      p.first->second = &tensor;
    }

    NodeArg* matching_graph_input = GetMutableMainFunction()->GetNodeArg(tensor.name());
    TypeProto t{TypeProtoFromTensorProto(tensor)};

    if (!utils::HasElemType(t.tensor_type())) {
      ORT_THROW("This is an invalid model. Tensor does not have type information.");
    }

    if (ir_version < 4) {
      // initializers can have matching graph inputs but are treated as constant,
      // so we prefer the shape from the initializer
      name_to_type_map[tensor.name()] = t;
      if (matching_graph_input != nullptr) {
        ORT_THROW_IF_ERROR(matching_graph_input->UpdateTypeAndShape(t, true, false, logger));
      }
    } else {
      // v4 and later allows a constant initializer with no matching graph input. create a NodeArg for these.
      // otherwise we prefer the shape from the graph input so leave matching_graph_input as is.
      if (matching_graph_input == nullptr) {
        name_to_type_map[tensor.name()] = t;
        ORT_IGNORE_RETURN_VALUE(GetMutableMainFunction()->GetOrCreateNodeArg(tensor.name(), &t));
      } else {
        LOGS(logger, WARNING) << "Initializer " << tensor.name()
                               << " appears in graph inputs and will not be treated as constant value/weight. "
                               << "This may prevent some of the graph optimizations, like const folding. "
                               << "Move it out of graph inputs if there is no need to override it, "
                               << "by either re-generating the model with latest exporter/converter "
                               << "or with the tool onnxruntime/tools/python/remove_initializer_from_input.py.";
      }
    }
  }

  for (auto& graph_output : graph_proto_->output()) {
    if (utils::HasName(graph_output) && utils::HasType(graph_output)) {
      auto& name = graph_output.name();
      name_to_type_map[name] = graph_output.type();
      // always create NodeArg for graph output, in case it's from initializer
      GetMutableMainFunction()->GetOrCreateNodeArg(name, &graph_output.type());
    }
  }

  for (auto& node_arg : graph_proto_->value_info()) {
    if (utils::HasName(node_arg) && utils::HasType(node_arg)) {
      name_to_type_map[node_arg.name()] = node_arg.type();
    }
  }

  for (const auto& node_proto : graph_proto_->node()) {
    GetMutableMainFunction()->AddNode(node_proto, name_to_type_map);
  }

}

}