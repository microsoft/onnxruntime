// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/function_utils.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/shape_inference/implementation.h"
#include "core/graph/function_impl.h"
#include "core/graph/model_load_utils.h"

namespace onnxruntime {
namespace function_utils {

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

struct Inliner {
  std::string prefix;
  const onnxruntime::NodeAttributes& attr_map;
  std::vector<InlinedHashMap<std::string, std::string>> rename_scopes;

  Inliner(std::string prefix_, const onnxruntime::NodeAttributes& attr_map_) : prefix(prefix_)
                                                                                   attr_map(attr_map_) {
    // Create an empty mapping for the top-level scope.
    inliner.rename_scopes.push_back();
  }

  void rename(std::string& name) {
    for (auto i = rename_scopes.size() - 1; i > 0; --i) {
      const auto& map = rename_scopes[i];
      auto iter = map.find(name);
      if (iter != map.end()) {
        name = iter->second;
        return;
      }
    }
    auto new_name = prefix + name;
    auto& current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  template <bool isOutput>
  void bind(const RepeatedPtrField<string>& formals, const RepeatedPtrField<string>& actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name AP.
    // However, if AP is empty, it is a missing optional parameter. This does not make any difference
    // for inputs. However, for outputs we use a unique dummy name to handle the case that it
    // is used in an output-context where it is not optional.
    ORT_ENFORCE(actuals.size() <= formals.size(),
                "Number of actual parameters cannot exceed number of formal parameters");
    auto& current_scope = rename_scopes.back();
    int i = 0;
    for (; i < actuals.size(); ++i) {
      std::string formal = formals.Get(i);
      std::string rename = actuals.Get(i);
      if (isOutput && rename.empty())
        rename = prefix + formal;
      current_scope[formal] = rename;
    }
    for (; i < formals.size(); ++i) {
      std::string formal = formals.Get(i);
      std::string rename = isOutput ? prefix + formal : std::string("");
      current_scope[formal] = rename;
    }
  }

  void transform(NodeProto& n) {
    auto& input = *n.mutable_input();
    for (auto it = input.begin(); it != input.end(); ++it) {
      *it = rename(*it);
    }
    auto& output = *n.mutable_output();
    for (auto it = output.begin(); it != output.end(); ++it) {
      *it = rename(*it);
    }
    auto& attributes = *n.mutable_attribute();
    for (auto* attr_iter = attributes.begin(); attr_iter != attributes.end(); ++attr_iter) {
      auto* attr = *attr_iter;
      if (!attr->ref_attr_name().empty()) {
        auto entry = attr_map.find(attr->ref_attr_name());
        if (entry != attr_map.cend()) {
          *attr = entry->second;
        } else {
          attr_iter = attributes.erase(attr_iter);
          continue;
        }
      }
      if (attr->has_g()) {
        transform(attr->mutable_g());
      }
      ++attr_iter
    }
  }

  void transform(GraphProto& graph) {
    for (auto& n : fp.mutable_node())
      transform(n);
  }

  static void createInlinableCopy(NodeProto& callnode, FunctionProto& callee) {
    std::string uniq_identifier = callnode.name();
    if (uniq_identifier.empty()) {
      std::stringstream ss;
      ss << static_cast<const void*>(&callnode);
      uniq_identifier = ss.str();
    }
    Inliner inliner(uniq_identifier, callnode.GetAttributes());

    inliner.bind(callee.input(), callnode.input());
    inliner.bind(callee.output(), callnode.output());

    for (auto& n : callee.mutable_node())
      inliner.transform(n);
  }
}

std::unique_ptr<ONNX_NAMESPACE::OpSchema>
CreateSchema(const Graph& graph,
             const IndexedSubGraph& nodes_to_fuse) {
  const auto* meta_def = nodes_to_fuse.GetMetaDef();
  auto op_schema = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema->SetName(meta_def->name);
  op_schema->SetDomain(meta_def->domain);
  op_schema->SetDoc(meta_def->doc_string);
  op_schema->SinceVersion(meta_def->since_version);

  if (meta_def->type_and_shape_inference_function) {
    op_schema->TypeAndShapeInferenceFunction(meta_def->type_and_shape_inference_function);
  }

  int i = 0;

  for (auto& input : meta_def->inputs) {
    auto input_arg = graph.GetNodeArg(input);
    // inputs must have a type. can be inferred for outputs.
    ORT_ENFORCE(input_arg->Type() != nullptr);
    op_schema->Input(i, input, "", *input_arg->Type());
    ++i;
  }
  i = 0;
  for (auto& output : meta_def->outputs) {
    auto output_arg = graph.GetNodeArg(output);
    op_schema->Output(i, output, "", *output_arg->Type());
    ++i;
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
  all_types.reserve(ONNX_NAMESPACE::OpSchema::all_tensor_types_with_bfloat().size() +
                    ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().size());
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_types_with_bfloat().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_types_with_bfloat().cend());
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cend());

  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  InlinedHashMap<std::string, int> opset_imports;
  for (const auto& relied_opset : onnx_func_proto.opset_import()) {
    opset_imports[relied_opset.domain()] = static_cast<int>(relied_opset.version());
  }

  std::function<void(const ONNX_NAMESPACE::NodeProto&)> process_node = [&](const ONNX_NAMESPACE::NodeProto& node) {
    auto it = opset_imports.find(node.domain());
    ORT_ENFORCE(it != opset_imports.end(), "No opset registered for domain " + node.domain() + " in function opset imports.");
    int domain_version = it->second;
    ORT_ENFORCE(domain_version != -1, "No opset registered for domain " + node.domain() + " in function opset imports.");
    const auto node_op_schema =
        schema_registry->GetSchema(node.op_type(), domain_version, node.domain());
    for (int i = 0; i < node.input_size(); ++i) {
      auto& in_name = node.input().Get(i);
      auto iter = input_name_idx_map.find(in_name);
      if (iter != input_name_idx_map.end()) {
        int idx = iter->second;
        std::string type_str = node_op_schema ? node_op_schema->inputs().at(i).GetTypeStr() + "in" + std::to_string(idx) : "Tin" + std::to_string(idx);
        input_types_list[idx] = std::make_pair(in_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          // If schema is available for the node then get the allowed types from the schema
          // else add all types to allowed types list. It is OK to add all types. Any issues will be
          // caught later if we try to inline the nodes and there is no kernl available for
          // the requested types.
          auto& dest_types = type_constraint_map[type_str];
          if (node_op_schema) {
            const auto& types = node_op_schema->inputs().at(i).GetTypes();
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
    }
    for (int i = 0; i < node.output_size(); ++i) {
      auto& out_name = node.output().Get(i);
      auto iter = output_name_idx_map.find(out_name);
      if (iter != output_name_idx_map.end()) {
        int idx = iter->second;
        std::string type_str = node_op_schema ? node_op_schema->outputs().at(i).GetTypeStr() + "out" + std::to_string(i) : "Tout" + std::to_string(i);
        output_types_list[idx] = std::make_pair(out_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          // If schema is available for the node then get the allowed types from the schema
          // else add all types to allowed types list. It is OK to add all types. Any issues will be
          // caught later if we try to inline the nodes and there is no kernel available for
          // the requested types.
          auto& dest_types = type_constraint_map[type_str];
          if (node_op_schema) {
            const auto& types = node_op_schema->outputs().at(i).GetTypes();
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
    op_schema->Input(i, input.first, "", input.second);
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
        std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> copy(model_local_functions.begin(), model_local_functions.end());
        ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(*onnx_func_proto, func_domain_to_version,
                                                                   schema_registry, ctx, options, copy);
      });

  op_schema->Finalize();
  return op_schema;
}

Status Instantiate(onnxruntime::Graph& graph,
                   const onnxruntime::NodeIndex node_index,
                   const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                   std::unique_ptr<Function>& output) {
  auto* node_in_parent_graph = graph.GetNode(node_index);
  ORT_ENFORCE(node_in_parent_graph);

  ONNX_NAMESPACE::NodeProto function_op_node_proto;  // NodeProto pertaining to the op with a FunctionBody
  node_in_parent_graph->ToProto(function_op_node_proto);

  output = std::make_unique<FunctionImpl>(graph, onnx_func_proto);
  auto& function_body_graph = output->MutableBody();

  // Create a copy of FunctionProto
  ONNX_NAMESPACE::FunctionProto inlined_fp = onnx_func_proto;
  Inliner::createInlinableCopy(*node_in_parent_graph, inlined_fp);

  // iterate over each node in the FunctionProto and fix inputs/outputs
  for (const auto* node : inlined_fp.node()) {
    InlinedVector<onnxruntime::NodeArg*> inputs;
    InlinedVector<onnxruntime::NodeArg*> outputs;

    for (int idx = 0; idx < (*node).input_size(); ++idx) {
      const std::string& tensor_name = (*node).input().Get(idx);
      auto& no_arg = function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr);
      inputs.push_back(&no_arg);
    }

    for (int idx = 0; idx < (*node).output_size(); ++idx) {
      std::string tensor_name = (*node).output().Get(idx);
      auto& no_arg = function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr);
      outputs.push_back(&no_arg);
    }

    onnxruntime::NodeAttributes new_attr_map;
    new_attr_map.reserve(node->attribute_size());
    for (const auto* node_attr = (*node).attribute()) {
      onnx::AttributeProto attr_copy = *node_attr;
      new_attr_map[(*node_attr).name()] = std::move(attr_copy);
    }
    function_body_graph.AddNode(uniq_identifier, node->op_type(),
                                node->doc_string(), inputs, outputs, &new_attr_map, node->domain());
  }

  std::vector<const NodeArg*> graph_inputs, graph_outputs;

  for (const auto& tensor_name : onnx_func_proto.input())
    graph_inputs.pushback(&function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr));

  for (const auto& tensor_name : onnx_func_proto.output())
    graph_outputs.pushback(&function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr));

  function_body_graph.SetInputs(graph_inputs);
  function_body_graph.SetOutputs(graph_outputs);

  onnxruntime::Graph::ResolveOptions options;
  ORT_RETURN_IF_ERROR(function_body_graph.Resolve(options));

  // ORT_RETURN_IF(node_in_parent_graph->InputDefs().size() != function_body_graph.GetInputsIncludingInitializers().size(),
  //   "Node " + node_in_parent_graph->Name() + "'s number of inputs is different from function body graph's number of input.");

  // ORT_RETURN_IF(node_in_parent_graph->OutputDefs().size() != function_body_graph.GetOutputs().size(),
  //             "Node ", node_in_parent_graph->Name(), "'s number of outputs is different from function body graph's number of outputs.");
  return Status::OK();
}

}  // namespace function_utils
}  // namespace onnxruntime
