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


// This method updates the names of inputs/outputs of nodes in subgraphs
// within nodes in an op that has a FunctionBody.
// Subgraphs within an op with a FunctionBody could be referencing inputs/outputs in the OpSchema
// and we need to replace these names with the corresponding input/output names from the actual model graph

// The arguments to this method are :
// (1) The 'subgraph' from a node containing it (ONNX::GraphProto)
// (2) The parent 'graph' - main model graph (OnnxRuntime::Graph)
// (3) The node with a function body (ONNX::NodeProto)
// (4) A map containing the input name from the op schema to the corresponding index
// E.g. For Range-11, {"start" : 0, "limit": 1, "delta": 2}
// (5) A map containing the output name from the op schema to the corresponding index
// E.g. For Range-11, {"output" : 0}
static void UpdateSubgraphsWithinFunctionBody(ONNX_NAMESPACE::GraphProto& subgraph_proto,
                                              const Graph& parent_graph,
                                              const ONNX_NAMESPACE::NodeProto& function_node_in_parent_graph,
                                              const InlinedHashMap<std::string, int>& input_name_idx_map,
                                              const InlinedHashMap<std::string, int>& output_name_idx_map) {
  // Iterate through all the nodes in the subgraph
  for (auto subgraph_node = subgraph_proto.mutable_node()->begin();
       subgraph_node != subgraph_proto.mutable_node()->end(); ++subgraph_node) {
    // Iterate through all the inputs of the current node
    for (int idx = 0; idx < (*subgraph_node).input_size(); ++idx) {
      const std::string& tensor_name = (*subgraph_node).input().Get(idx);
      auto iter = input_name_idx_map.find(tensor_name);
      // If an input pertaining to the name in the op schema is found,
      // replace it with the corresponding input to the node with function body from the actual model graph
      if (iter != input_name_idx_map.end()) {
        const auto parent_graph_input_to_function_node = function_node_in_parent_graph.input().Get(iter->second);
        (*subgraph_node).set_input(idx, parent_graph_input_to_function_node);
      }
    }
    // Iterate through all the output of the current node
    for (int idx = 0; idx < (*subgraph_node).output_size(); ++idx) {
      const std::string& tensor_name = (*subgraph_node).output().Get(idx);
      auto iter = output_name_idx_map.find(tensor_name);
      if (iter != output_name_idx_map.end()) {
        // If an input pertaining to the name in the op schema is found,
        // replace it with the corresponding output to the node with function body from the actual model graph
        const auto& parent_graph_output_to_function_node = function_node_in_parent_graph.output().Get(iter->second);
        (*subgraph_node).set_output(idx, parent_graph_output_to_function_node);
      }
    }

    for (auto subgraph_node_attr = (*subgraph_node).mutable_attribute()->begin();
         subgraph_node_attr != (*subgraph_node).mutable_attribute()->end(); ++subgraph_node_attr) {
      if ((*subgraph_node_attr).has_f()) {
        ORT_THROW(
            "A node with a function body within a subgraph within another function body "
            "is currently not supported in ORT");
      }
      // Recurse into any subgraphs in the current subgraph being processed
      if ((*subgraph_node_attr).has_g()) {
        UpdateSubgraphsWithinFunctionBody(*(*subgraph_node_attr).mutable_g(),
                                          parent_graph, function_node_in_parent_graph,
                                          input_name_idx_map, output_name_idx_map);
      }
    }
  }
}

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
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
  for (const auto& node : onnx_func_proto.node()) {
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
    }
  }

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
        std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*> map_copy(model_local_functions.begin(),
                                                                                       model_local_functions.end());
        ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(*onnx_func_proto, func_domain_to_version,
                                                                   schema_registry, ctx, options, map_copy);
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
  std::vector<const NodeArg*> graph_inputs(node_in_parent_graph->InputDefs().size(), nullptr),
      graph_outputs(node_in_parent_graph->OutputDefs().size(), nullptr);

  // Add node and node args into subgraph
  // The subgraph preserved the input/output tensor names
  // in the parent graph for later inlining purpose
  const auto& attr_map = node_in_parent_graph->GetAttributes();

  ONNX_NAMESPACE::NodeProto function_op_node_proto;  // NodeProto pertaining to the op with a FunctionBody
  node_in_parent_graph->ToProto(function_op_node_proto);

  InlinedHashSet<std::string_view> node_input_outputs;
  auto parent_input_defs = node_in_parent_graph->InputDefs();
  auto parent_output_defs = node_in_parent_graph->OutputDefs();
  node_input_outputs.reserve(parent_input_defs.size() + parent_output_defs.size());

  for (const auto* input_def : parent_input_defs) {
    if (input_def->Exists()) {
      node_input_outputs.insert(input_def->Name());
    }
  }

  for (const auto* output_def : parent_output_defs) {
    if (output_def->Exists()) {
      node_input_outputs.insert(output_def->Name());
    }
  }

  ONNX_NAMESPACE::TypeProto tensor_int32;  // dummy type used for unused formal parameters
  tensor_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  output = std::make_unique<FunctionImpl>(graph, onnx_func_proto);

  auto& function_body_graph = output->MutableBody();
  InlinedHashMap<std::string, int> input_name_idx_map;
  InlinedHashMap<std::string, int> output_name_idx_map;
  InlinedHashMap<std::string, std::string> internal_input_output_updates;

  for (int i = 0; i < onnx_func_proto.input_size(); ++i) {
    input_name_idx_map[onnx_func_proto.input().Get(i)] = i;
  }
  for (int i = 0; i < onnx_func_proto.output_size(); ++i) {
    output_name_idx_map[onnx_func_proto.output().Get(i)] = i;
  }
  // iterate over each node in the FunctionProto and fix inputs/outputs
  for (auto node = onnx_func_proto.node().begin(); node != onnx_func_proto.node().end(); ++node) {
    InlinedVector<onnxruntime::NodeArg*> inputs;
    InlinedVector<onnxruntime::NodeArg*> outputs;
    std::string uniq_identifier = (*node).name();
    if (!utils::HasName(*node)) {
      std::stringstream ss;
      ss << static_cast<const void*>(&(*node));
      uniq_identifier = ss.str();
    }

    for (int idx = 0; idx < (*node).input_size(); ++idx) {
      const std::string& tensor_name = (*node).input().Get(idx);
      if (tensor_name.empty()) {
        auto& no_arg = function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr);
        inputs.push_back(&no_arg);
        continue;
      }
      auto iter = input_name_idx_map.find(tensor_name);
      if (iter != input_name_idx_map.end()) {
        // If input is part of function inputs, preserve NodeArg and input/output names
        const std::string& actual_parameter_name = function_op_node_proto.input().Get(iter->second);
        if (!actual_parameter_name.empty()) {
          const onnxruntime::NodeArg* node_arg = graph.GetNodeArg(actual_parameter_name);
          const ONNX_NAMESPACE::TypeProto* actual_type = node_arg->TypeAsProto();
          auto& n_input = function_body_graph.GetOrCreateNodeArg(actual_parameter_name, actual_type);
          inputs.push_back(&n_input);
          graph_inputs[iter->second] = &n_input;
        } else {
          // Unused optional parameter to function
          auto& n_input = function_body_graph.GetOrCreateNodeArg(actual_parameter_name, nullptr);
          inputs.push_back(&n_input);
          auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
          graph_inputs[iter->second] = &unused_formal_param;
        }
      } else {
        // If input is part of function outputs, preserve NodeArg and input/output names
        iter = output_name_idx_map.find(tensor_name);
        if (iter != output_name_idx_map.end()) {
          const std::string& actual_parameter_name = function_op_node_proto.output().Get(iter->second);
          const onnxruntime::NodeArg* node_arg = graph.GetNodeArg(actual_parameter_name);
          const ONNX_NAMESPACE::TypeProto* actual_type = node_arg->TypeAsProto();
          auto& n_input = function_body_graph.GetOrCreateNodeArg(actual_parameter_name, actual_type);
          inputs.push_back(&n_input);
        } else {
          // Input is intermediate input in function body.
          // Check if input name needs to be mapped to a new unique name (this is required when node input\outputs
          // have same names as intermediate input\outputs.
          auto it = internal_input_output_updates.find(tensor_name);
          if (it != internal_input_output_updates.end()) {
            auto& n_input = function_body_graph.GetOrCreateNodeArg(
                it->second, nullptr);
            inputs.push_back(&n_input);
          } else {
            // Input is intermediate function body input and has no name collision with node input\output
            // It can be added to the graph without any modification
            auto& n_input = function_body_graph.GetOrCreateNodeArg(
                tensor_name, nullptr);
            inputs.push_back(&n_input);
          }
        }
      }
    }

    for (int idx = 0; idx < (*node).output_size(); ++idx) {
      std::string tensor_name = (*node).output().Get(idx);
      if (tensor_name.empty()) {
        auto& no_arg = function_body_graph.GetOrCreateNodeArg(tensor_name, nullptr);
        outputs.push_back(&no_arg);
        continue;
      }
      auto iter = output_name_idx_map.find(tensor_name);
      if (iter != output_name_idx_map.end()) {
        // Preserving NodeArg and input/output names
        const std::string& actual_parameter_name = function_op_node_proto.output().Get(iter->second);
        if (!actual_parameter_name.empty()) {
          const onnxruntime::NodeArg* node_arg = graph.GetNodeArg(actual_parameter_name);
          const ONNX_NAMESPACE::TypeProto* actual_type = node_arg->TypeAsProto();
          auto& n_output = function_body_graph.GetOrCreateNodeArg(actual_parameter_name, actual_type);
          outputs.push_back(&n_output);
          graph_outputs[iter->second] = &n_output;
        } else {
          // Unused optional parameter to function
          auto& n_output = function_body_graph.GetOrCreateNodeArg(actual_parameter_name, nullptr);
          outputs.push_back(&n_output);
          auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
          graph_outputs[iter->second] = &unused_formal_param;
        }
      } else {
        // Output is intermediate output in function body.
        // Check if output name needs to be mapped to a new unique name (this is required when node input\outputs
        // have same names as intermediate input\outputs.
        auto it = node_input_outputs.find(tensor_name);
        if (it != node_input_outputs.end()) {
          auto& n_output = function_body_graph.GetOrCreateNodeArg(
              tensor_name + uniq_identifier, nullptr);
          outputs.push_back(&n_output);
          internal_input_output_updates.insert({tensor_name, tensor_name + uniq_identifier});
        } else {
          auto& n_output = function_body_graph.GetOrCreateNodeArg(
              tensor_name, nullptr);
          outputs.push_back(&n_output);
        }
      }
    }

    // Formal parameters unused in function body. For now, we retain them in the graph's input and
    // output list (with a dummy type).
    // TODO: Need a proper scheme to generate unique names to avoid name-collision.
    for (unsigned i = 0; i < graph_inputs.size(); ++i) {
      if (graph_inputs[i] == nullptr) {
        auto tensor_name = onnx_func_proto.input(i) + "_dummy";
        auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
        graph_inputs[i] = &unused_formal_param;
      }
    }

    for (unsigned i = 0; i < graph_outputs.size(); ++i) {
      if (graph_outputs[i] == nullptr) {
        auto tensor_name = onnx_func_proto.output(i) + "_dummy";
        auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
        graph_outputs[i] = &unused_formal_param;
      }
    }

    onnxruntime::NodeAttributes new_attr_map;
    new_attr_map.reserve(node->attribute_size());
    for (auto node_attr = (*node).attribute().begin();
         node_attr != (*node).attribute().end(); ++node_attr) {
      if (!(*node_attr).ref_attr_name().empty()) {
        auto entry = attr_map.find((*node_attr).ref_attr_name());
        if (entry != attr_map.cend()) {
          onnx::AttributeProto attr_copy = entry->second;
          attr_copy.set_name(node_attr->name());
          new_attr_map[(*node_attr).name()] = std::move(attr_copy);
        }
      } else {
        onnx::AttributeProto attr_copy = *node_attr;
        // If this node contains subgraphs, the node inputs/outputs within them needs to be fixed as well
        if ((*node_attr).has_g()) {
          UpdateSubgraphsWithinFunctionBody(*attr_copy.mutable_g(),
                                            graph, function_op_node_proto,
                                            input_name_idx_map, output_name_idx_map);
        }
        new_attr_map[(*node_attr).name()] = std::move(attr_copy);
      }
    }
    function_body_graph.AddNode(uniq_identifier, node->op_type(),
                                node->doc_string(), inputs, outputs, &new_attr_map, node->domain());
  }

  function_body_graph.SetInputs(graph_inputs);
  function_body_graph.SetOutputs(graph_outputs);

  onnxruntime::Graph::ResolveOptions options;
  ORT_RETURN_IF_ERROR(function_body_graph.Resolve(options));

  ORT_RETURN_IF(node_in_parent_graph->InputDefs().size() != function_body_graph.GetInputsIncludingInitializers().size(),
    "Node " + node_in_parent_graph->Name() + "'s number of inputs is different from function body graph's number of input.");

  ORT_RETURN_IF(node_in_parent_graph->OutputDefs().size() != function_body_graph.GetOutputs().size(),
              "Node ", node_in_parent_graph->Name(), "'s number of outputs is different from function body graph's number of outputs.");
  return Status::OK();
}

}
}
