// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>

#include "core/graph/function_utils.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/shape_inference/implementation.h"
#include "core/graph/function_impl.h"
#include "core/graph/model_load_utils.h"

namespace onnxruntime {
namespace function_utils {

using string = std::string;
using namespace ONNX_NAMESPACE;

// Utilify function to get the imported version of domain from opset imports
// Returns -1 if requested domain is not found in the opset_imports
template <template <typename, typename> class M>
static int GetVersionForDomain(const std::string& domain, const M<std::string, int>& opset_imports) {
  auto it = opset_imports.find(domain);
  if (it == opset_imports.end()) {
    return -1;
  }
  return it->second;
}

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(
    const Graph& graph,
    const IndexedSubGraph& nodes_to_fuse, bool allow_aggregated_tensor_type) {
  const auto* meta_def = nodes_to_fuse.GetMetaDef();

  using ONNX_NAMESPACE::OpSchema;
  auto op_schema = std::make_unique<OpSchema>(meta_def->name, __FILE__, __LINE__);
  op_schema->SetDomain(meta_def->domain);
  op_schema->SetDoc(meta_def->doc_string);
  op_schema->SinceVersion(meta_def->since_version);

  if (meta_def->type_and_shape_inference_function) {
    op_schema->TypeAndShapeInferenceFunction(meta_def->type_and_shape_inference_function);
  }

  if (allow_aggregated_tensor_type) {
    // The generated schema will use the same type constraint for all inputs and outputs,
    // and that type constraint will match all tensor types.
    // Due to this, a user of this style of schema must manually check whether any applicable type constraints
    // for each input or output are satisfied prior to creating a node that uses this schema
    //
    op_schema->TypeConstraint("TAggregatedTypes", ONNX_NAMESPACE::OpSchema::all_tensor_types_ir4(),
                              "all_tensor_types_ir4");
  }

  int i = 0;
  for (const auto& input : meta_def->inputs) {
    const auto* input_arg = graph.GetNodeArg(input);
    // inputs must have a type. can be inferred for outputs.
    ORT_ENFORCE(input_arg->Type() != nullptr);
    op_schema->Input(i, input, "",
                     allow_aggregated_tensor_type ? "TAggregatedTypes" : *input_arg->Type(),
                     OpSchema::FormalParameterOption::Single, /*is_homogeneous=*/!allow_aggregated_tensor_type);
    i++;
  }

  i = 0;
  for (const auto& output : meta_def->outputs) {
    const auto* output_arg = graph.GetNodeArg(output);
    op_schema->Output(i, output, "",
                      allow_aggregated_tensor_type ? "TAggregatedTypes" : *output_arg->Type(),
                      OpSchema::FormalParameterOption::Single, /*is_homogeneous=*/!allow_aggregated_tensor_type);
    i++;
  }
  op_schema->Finalize();

  return op_schema;
}

// Auto inferred and generate an opschema for stand-alone functions
// TODO: revisit to see if we can eliminate typeconstraint step
static void IOTypeConstraintHelper(const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                                   std::unique_ptr<ONNX_NAMESPACE::OpSchema>& op_schema,
                                   const InlinedHashMap<std::string, int>& input_name_idx_map,
                                   const InlinedHashMap<std::string, int>& output_name_idx_map) {
  std::vector<std::pair<std::string, std::string>> input_types_list(onnx_func_proto.input_size());
  std::vector<std::pair<std::string, std::string>> output_types_list(onnx_func_proto.output_size());

  size_t num_of_inputs = 0;
  size_t num_of_outputs = 0;
  for (const auto& node : onnx_func_proto.node()) {
    num_of_inputs += node.input_size();
    num_of_outputs += node.output_size();
  }

  InlinedHashMap<std::string, std::vector<std::string>> type_constraint_map;
  type_constraint_map.reserve(num_of_inputs + num_of_outputs);
  InlinedHashMap<std::string_view, ONNX_NAMESPACE::AttributeProto_AttributeType> attribute_type_map;
  attribute_type_map.reserve(onnx_func_proto.node_size());

  // Create an all permissive list of data types. This will be used in case of model local functions
  // when we cannot infer the type constraints from function proto body
  InlinedHashSet<std::string_view> all_types;
  all_types.reserve(ONNX_NAMESPACE::OpSchema::all_tensor_types_ir4().size() +
                    ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().size());
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_types_ir4().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_types_ir4().cend());
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cend());

  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  InlinedHashMap<std::string, int> opset_imports;
  for (const auto& relied_opset : onnx_func_proto.opset_import()) {
    opset_imports[relied_opset.domain()] = static_cast<int>(relied_opset.version());
  }

  std::function<void(const ONNX_NAMESPACE::NodeProto&)> process_node = [&](const ONNX_NAMESPACE::NodeProto& node) {
    auto it = opset_imports.find(node.domain());
    ORT_ENFORCE(it != opset_imports.end(),
                "No opset registered for domain " + node.domain() + " in function opset imports.");
    int domain_version = it->second;
    ORT_ENFORCE(domain_version != -1,
                "No opset registered for domain " + node.domain() + " in function opset imports.");

    const auto* node_op_schema = schema_registry->GetSchema(node.op_type(), domain_version, node.domain());
    int variadic_arg_idx = -1;
    for (int i = 0; i < node.input_size(); ++i) {
      if (node_op_schema && variadic_arg_idx == -1) {
        // The check is applied only if we have not seen a variadic parameter so far:
        ORT_ENFORCE(static_cast<size_t>(i) < node_op_schema->inputs().size(),
                    "Too many inputs for op " + node.op_type());
      }

      auto& in_name = node.input().Get(i);
      auto iter = input_name_idx_map.find(in_name);
      if (iter != input_name_idx_map.end()) {
        int idx = iter->second;
        // if we have hit a variadic arg it is the last input in the schema, so we need to use that index not i.
        auto schema_idx = variadic_arg_idx != -1 ? variadic_arg_idx : i;

        std::string type_str;
        if (node_op_schema) {
          type_str = node_op_schema->inputs().at(schema_idx).GetTypeStr() + "in" + std::to_string(idx);
        } else {
          type_str = "Tin" + std::to_string(idx);
        }

        input_types_list[idx] = std::make_pair(in_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          // If schema is available for the node then get the allowed types from the schema
          // else add all types to allowed types list. It is OK to add all types. Any issues will be
          // caught later if we try to inline the nodes and there is no kernl available for
          // the requested types.
          auto& dest_types = type_constraint_map[type_str];
          if (node_op_schema) {
            const auto& types = node_op_schema->inputs().at(schema_idx).GetTypes();
            dest_types.reserve(dest_types.size() + types.size());
            for (const auto* s : types) {
              dest_types.emplace_back(*s);
            }
          } else {
            dest_types.reserve(dest_types.size() + all_types.size());
            for (const auto& s : all_types) {
              dest_types.emplace_back(s);
            }
          }
        }
      }

      // if this is a variadic input there are no more inputs in the schema
      if (node_op_schema && variadic_arg_idx == -1 &&
          node_op_schema->inputs().at(i).GetOption() == OpSchema::FormalParameterOption::Variadic) {
        variadic_arg_idx = i;
      }
    }

    variadic_arg_idx = -1;
    for (int i = 0; i < node.output_size(); ++i) {
      auto& out_name = node.output().Get(i);
      auto iter = output_name_idx_map.find(out_name);
      if (iter != output_name_idx_map.end()) {
        int idx = iter->second;
        // if we have hit a variadic arg it is the last output in the schema, so we need to use that index.
        auto schema_idx = variadic_arg_idx != -1 ? variadic_arg_idx : i;

        std::string type_str;
        if (node_op_schema) {
          type_str = node_op_schema->outputs().at(schema_idx).GetTypeStr() + "out" + std::to_string(idx);
        } else {
          type_str = "Tout" + std::to_string(idx);
        }

        output_types_list[idx] = std::make_pair(out_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          // If schema is available for the node then get the allowed types from the schema
          // else add all types to allowed types list. It is OK to add all types. Any issues will be
          // caught later if we try to inline the nodes and there is no kernel available for
          // the requested types.
          auto& dest_types = type_constraint_map[type_str];
          if (node_op_schema) {
            const auto& types = node_op_schema->outputs().at(schema_idx).GetTypes();
            dest_types.reserve(dest_types.size() + types.size());
            for (auto* data_type : types) {
              dest_types.emplace_back(*data_type);
            }
          } else {
            dest_types.reserve(dest_types.size() + all_types.size());
            for (const auto& data_type : all_types) {
              dest_types.emplace_back(data_type);
            }
          }
        }
      }

      // if this is a variadic output there are no more outputs in the schema
      if (node_op_schema && variadic_arg_idx == -1 &&
          node_op_schema->outputs().at(i).GetOption() == OpSchema::FormalParameterOption::Variadic) {
        variadic_arg_idx = i;
      }
    }

    // If an subgraph node attribute has a specified
    // type attribute, we add its referenced attribute
    // into the op's schema
    for (auto& attr : node.attribute()) {
      if (!attr.ref_attr_name().empty() && utils::HasType(attr))
        attribute_type_map[attr.ref_attr_name()] = attr.type();
      if (attr.ref_attr_name().empty() && attr.has_g()) {
        for (const auto& sgnode : attr.g().node())
          process_node(sgnode);
      }
    }
  };

  for (const auto& node : onnx_func_proto.node())
    process_node(node);

  int i = 0;
  for (auto& input : input_types_list) {
    if (!input.first.empty()) {
      op_schema->Input(i, input.first, "", input.second);
    } else {
      // Handle unused input: its type can be anything.
      std::string type_str = "Tin" + std::to_string(i);
      op_schema->Input(i, onnx_func_proto.input(i), "", type_str);
      auto& dest_types = type_constraint_map[type_str];
      dest_types.reserve(dest_types.size() + all_types.size());
      for (const auto& data_type : all_types) {
        dest_types.emplace_back(data_type);
      }
    }
    ++i;
  }
  i = 0;
  for (auto& output : output_types_list) {
    op_schema->Output(i, output.first, "", output.second);
    ++i;
  }

  for (auto& tc : type_constraint_map) {
    op_schema->TypeConstraint(tc.first, tc.second, "");
  }

  for (auto& attribute_name : onnx_func_proto.attribute()) {
    if (attribute_type_map.count(attribute_name))
      op_schema->Attr(attribute_name, "", attribute_type_map[attribute_name], false);
  }
}

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const std::string& function_domain,
                                                       const std::string& function_name,
                                                       const InlinedHashMap<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
                                                       const std::unordered_map<std::string, int>& domain_version_map,
                                                       const SchemaRegistryManager& schema_registry,
                                                       const logging::Logger& logger,
                                                       bool allow_released_opsets_only) {
  std::string func_identifier = function_utils::GetFunctionIdentifier(function_domain, function_name);
  auto iter = model_local_functions.find(func_identifier);
  if (iter == model_local_functions.end()) {
    ORT_THROW("The given function name: ", function_name, ", domain: ", function_domain, " is not found in model local functions");
  }

  auto* onnx_func_proto = iter->second;
  ORT_ENFORCE(onnx_func_proto);
  // generate the schema for this function template
  // For schema defined functions get the version from the node in parent graph.
  // For the functions which do not have schema defined (model local functions)
  // get the since version from the version in opset imports using the domain.
  auto it = domain_version_map.find(function_domain);
  auto since_version = it == domain_version_map.end() ? -1 : it->second;
  auto op_schema = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema->SetName(function_name);
  op_schema->SetDomain(function_domain);
  op_schema->SetDoc(onnx_func_proto->doc_string());
  op_schema->SinceVersion(static_cast<ONNX_NAMESPACE::OperatorSetVersion>(since_version));
  InlinedHashMap<std::string, int> input_name_idx_map;
  InlinedHashMap<std::string, int> output_name_idx_map;

  for (int i = 0; i < onnx_func_proto->input_size(); ++i) {
    input_name_idx_map[onnx_func_proto->input().Get(i)] = i;
  }
  for (int i = 0; i < onnx_func_proto->output_size(); ++i) {
    output_name_idx_map[onnx_func_proto->output().Get(i)] = i;
  }

  // Infer a op_schema for stand-alone functions.
  IOTypeConstraintHelper(*onnx_func_proto, op_schema, input_name_idx_map, output_name_idx_map);
  auto allow_official_onnx_release_only_final =
      allow_released_opsets_only && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  const auto onnx_released_versions =
      schema_registry.GetLastReleasedOpsetVersions(false);

  std::unordered_map<std::string, int> func_domain_to_version;
  for (auto& opSet : onnx_func_proto->opset_import()) {
    const auto& domain = opSet.domain();
    const auto version = gsl::narrow_cast<int>(opSet.version());

    model_load_utils::ValidateOpsetForDomain(onnx_released_versions, logger,
                                             allow_official_onnx_release_only_final, domain, version);

    // We need to overwrite the domain here with ("") or else the loop below will try to find ("")
    // in the map and if not found (when domain == kOnnxDomainAlias), adds an entry for ("", 11).
    // This effectively ignores the opset version specified by the model for the onnx domain.
    if (domain == kOnnxDomainAlias) {
      func_domain_to_version[kOnnxDomain] = version;
    } else {
      func_domain_to_version[domain] = version;
    }
  }

  op_schema->TypeAndShapeInferenceFunction(
      [onnx_func_proto, func_domain_to_version, &model_local_functions](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        ONNX_NAMESPACE::ShapeInferenceOptions options{true, 1, false};
        std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> map_copy(model_local_functions.begin(),
                                                                                       model_local_functions.end());
        std::unordered_map<std::string, TensorShapeProto> empty_map;
        ONNX_NAMESPACE::shape_inference::SymbolTableImpl symbolTable;
        ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(*onnx_func_proto, func_domain_to_version,
                                                                   schema_registry, ctx, options, map_copy,
                                                                   &symbolTable, &empty_map);
      });

  op_schema->Finalize();
  return op_schema;
}

class Inliner {
 private:
  std::string prefix;
  const onnxruntime::NodeAttributes& attr_map;
  std::vector<InlinedHashMap<std::string, std::string>> rename_scopes;

  Inliner(std::string prefix_, const onnxruntime::NodeAttributes& attr_map_) : prefix(prefix_),
                                                                               attr_map(attr_map_) {
    // Create an empty mapping for the top-level scope.
    rename_scopes.emplace_back();
  }

  // Replace given name with a unique version of the name, and cache the
  // renaming-binding in current scope.
  void make_unique(std::string& name) {
    auto new_name = prefix + name;
    auto& current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  void rename(std::string& name, bool is_new_def) {
    if (name.empty()) return;
    for (auto i = rename_scopes.size(); i > 0; --i) {
      const auto& map = rename_scopes[i - 1];
      auto iter = map.find(name);
      if (iter != map.end()) {
        name = iter->second;
        return;
      }
    }
    if (is_new_def) {
      make_unique(name);
    }
    // Otherwise, it is a reference to an outer-scope variable that should not be renamed.
  }

  template <bool isOutput>
  void bind(google::protobuf::RepeatedPtrField<string>& formals, const google::protobuf::RepeatedPtrField<string>& actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name AP.
    // However, if AP is empty, it is a missing optional parameter. This does not make any difference
    // for inputs. However, for outputs we use a unique dummy name to handle the case that it
    // is used in an output-context where it is not optional.
    ORT_ENFORCE(actuals.size() <= formals.size(),
                "Number of actual parameters cannot exceed number of formal parameters");
    auto& current_scope = rename_scopes.back();
    int i = 0;
    for (; i < actuals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = actuals.Get(i);
      if constexpr (isOutput)
        if (rename_as.empty())
          rename_as = prefix + formal;
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
    for (; i < formals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = isOutput ? prefix + formal : std::string("");
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
  }

  // Process a node:
  void transform(NodeProto& n) {
    if (!n.name().empty())
      n.set_name(prefix + n.name());

    for (auto& x : *n.mutable_input()) {
      rename(x, false);
    }
    for (auto& y : *n.mutable_output()) {
      rename(y, true);
    }
    auto& attributes = *n.mutable_attribute();
    for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
      auto& attr = *attr_iter;
      if (!attr.ref_attr_name().empty()) {
        // Attribute-references must be replaced by the corresponding attribute-value in the call-node
        // if the call-node contains the attribute. Otherwise, this attribute must be removed.
        auto entry = attr_map.find(attr.ref_attr_name());
        if (entry != attr_map.cend()) {
          // Copy value of attribute, but retain original name:
          std::string name = attr.name();
          attr = entry->second;
          attr.set_name(name);
        } else {
          attr_iter = attributes.erase(attr_iter);
          continue;
        }
      }
      // Subgraphs must be recursively processed.
      if (attr.has_g()) {
        transform(*attr.mutable_g());
      }
      for (auto& graph : *attr.mutable_graphs())
        transform(graph);
      ++attr_iter;
    }
  }

  // Process a sub-graph, contained as an attribute in a control-flow op node.
  void transform(GraphProto& graph) {
    rename_scopes.emplace_back();
    for (auto& x : *graph.mutable_input())
      make_unique(*x.mutable_name());
    for (auto& init : *graph.mutable_initializer())
      make_unique(*init.mutable_name());
    for (auto& y : *graph.mutable_output())
      make_unique(*y.mutable_name());
    for (auto& n : *graph.mutable_node())
      transform(n);
    rename_scopes.pop_back();
  }

 public:
  // The main specialization method: specialize a FunctionProto for a particular call-site.
  static void specialize(const NodeProto& callnode, FunctionProto& callee, const onnxruntime::NodeAttributes& attr_map, std::string unique_prefix) {
    Inliner inliner(unique_prefix, attr_map);

    inliner.bind<false>(*callee.mutable_input(), callnode.input());
    inliner.bind<true>(*callee.mutable_output(), callnode.output());

    for (auto& n : *callee.mutable_node())
      inliner.transform(n);
  }
};

void Specialize(ONNX_NAMESPACE::FunctionProto& called_function, const ONNX_NAMESPACE::NodeProto calling_node,
                const onnxruntime::NodeAttributes& attr_map, std::string unique_prefix) {
  Inliner::specialize(calling_node, called_function, attr_map, unique_prefix);
}

void Specialize(ONNX_NAMESPACE::FunctionProto& called_function, Node& calling_node, std::string unique_prefix) {
  ONNX_NAMESPACE::NodeProto calling_node_proto;
  calling_node.ToProto(calling_node_proto);

  onnxruntime::NodeAttributes attr_map = calling_node.GetAttributes();
  for (auto& attribute_proto : called_function.attribute_proto()) {
    auto entry = attr_map.find(attribute_proto.name());
    if (entry == attr_map.cend()) {
      attr_map[attribute_proto.name()] = attribute_proto;
    }
  }
  Specialize(called_function, calling_node_proto, attr_map, unique_prefix);
}

}  // namespace function_utils
}  // namespace onnxruntime
