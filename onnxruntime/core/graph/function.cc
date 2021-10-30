// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
// Infer shape for functions. This also supports 
// nested model local functions.
// TODO: Add this to onnx instead of adding it here.
void InferShapeForFunctionNode(
    InferenceContext& ctx,
    const FunctionProto& func_proto,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ShapeInferenceOptions& options,
    const ISchemaRegistry* schema_registry,
    const std::unordered_map<std::string, const FunctionProto*>& in_model_functions,
    std::function<std::string(const std::string& function_domain, const std::string& function_name)> get_func_id) {
  GraphProto g;
  // Get a temporary tensor-shape map
  const auto num_func_inputs = func_proto.input_size();
  std::unordered_map<std::string, TypeProto*> value_types_by_name;
  std::vector<TypeProto> types_cache(func_proto.input_size());
  for (int i = 0; i < num_func_inputs; ++i) {
    types_cache[i] = *ctx.getInputType(i);
    value_types_by_name[func_proto.input().Get(i)] = &types_cache[i];
  }

  // Get a temporary initial value map
  std::unordered_map<std::string, const TensorProto*> initializers_by_name;
  std::unordered_map<std::string, const SparseTensorProto*> sparse_initializers_by_name;
  for (int i = 0; i < static_cast<int>(ctx.getNumInputs()) && i < num_func_inputs; ++i) {
    const TypeProto* type = ctx.getInputType(i);
    if (type->value_case() == TypeProto::kTensorType && ctx.getInputData(i) != nullptr) {
      initializers_by_name[func_proto.input().Get(i)] = ctx.getInputData(i);
    } else if (type->value_case() == TypeProto::kSparseTensorType &&
               ctx.getInputSparseData(i) != nullptr) {
      sparse_initializers_by_name[func_proto.input().Get(i)] = ctx.getInputSparseData(i);
    }
  }
  std::unordered_map<std::string, const AttributeProto*> attr_map;
  for (auto& attr : func_proto.attribute()) {
    if (ctx.getAttribute(attr) != nullptr) {
      attr_map[attr] = ctx.getAttribute(attr);
    }
  }

  for (auto& n : func_proto.node()) {
    NodeProto copy_n(n);
    // Add attribute information into the temporary node
    copy_n.clear_attribute();
    for (const auto& attr : n.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          auto copy_attr = *attr_map[attr.ref_attr_name()];
          copy_attr.set_name(attr.name());
          copy_n.add_attribute()->CopyFrom(copy_attr);
        }
      } else {
        copy_n.add_attribute()->CopyFrom(attr);
      }
    }
    ONNX_NAMESPACE::shape_inference::InferenceContextImpl func_node_ctx(
        copy_n, value_types_by_name, initializers_by_name, sparse_initializers_by_name, {});

    // Resolve domain for node
    auto it = func_opset_imports.find(n.domain());
    if (it == func_opset_imports.end()) {
      fail_type_inference("Cannot infer type and shape for function", func_proto.name(),
                          ". No opset import for domain", n.domain(), " referenced by function body node ",
                          n.name(), " optype ", n.op_type());
    }
    auto domain_version = it->second;
    const auto schema = schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    if (schema) {
      schema->GetTypeAndShapeInferenceFunction()(func_node_ctx);
    } else {
      // check model local functions for FunctionProto
      auto iter = in_model_functions.find(get_func_id(n.domain(), n.op_type()));
      if (iter == in_model_functions.end()) {
        return;
      }

      std::unordered_map<std::string, int> func_node_opset_imports;
      for (const auto& opset_import : iter->second->opset_import()) {
        // If graph imports does not contain opset_import then insert it otherwise the one in graph imports overrides.
        // If the opset imports are not compatible then this will be caught during function body inline.
        func_node_opset_imports.insert({opset_import.domain(), static_cast<int>(opset_import.version())});
      }

      InferShapeForFunctionNode(func_node_ctx, *iter->second, func_node_opset_imports, options, schema_registry, in_model_functions, get_func_id);
    }

    for (int i = 0; i < copy_n.output_size(); ++i) {
      TypeProto* inferred_output_type = func_node_ctx.getOutputType(i);
      // Checking, Storing the inferred information
      auto iter = value_types_by_name.find(n.output(i));
      TypeProto* existingType = nullptr;
      if (iter != value_types_by_name.end()) {
        existingType = iter->second;
        shape_inference::checkShapesAndTypes(*inferred_output_type, *existingType);
      } else {
        // Store the inferred type info in the
        // subgraph temporarily
        auto vi = g.add_value_info();
        vi->set_name(copy_n.output(i));
        existingType = vi->mutable_type();
      }

      shape_inference::mergeShapesAndTypes(*inferred_output_type, existingType);

      // Make merged info available to further inference.
      value_types_by_name[copy_n.output(i)] = existingType;
    }
  }
  for (int i = 0; i < func_proto.output_size(); ++i) {
    const std::string& output_name = func_proto.output().Get(i);
    // Skip if no type inferred for the tensor
    auto iter = value_types_by_name.find(output_name);
    if (iter != value_types_by_name.cend()) {
      // Copy the type info to ctx
      // to pass back to main graph
      auto type_proto = ctx.getOutputType(i);
      type_proto->CopyFrom(*(iter->second));
    }
  }
}

}  // namespace ONNX_NAMESPACE
namespace onnxruntime {

// Utilify function to get the imported version of domain from opset imports
// Returns -1 if requested domain is not found in the opset_imports
static int GetVersionForDomain(const std::string& domain, const std::unordered_map<std::string, int>& opset_imports) {
  auto it = opset_imports.find(domain);
  if (it == opset_imports.end()) {
    return -1;
  }
  return it->second;
}

// Auto inferred and generate an opschema for stand-alone functions
// TODO: revisit to see if we can eliminate typeconstraint step
void IOTypeConstraintHelper(const ONNX_NAMESPACE::FunctionProto& onnx_func_proto_,
                            std::unique_ptr<ONNX_NAMESPACE::OpSchema>& op_schema_,
                            const std::unordered_map<std::string, int>& input_name_idx_map,
                            const std::unordered_map<std::string, int>& output_name_idx_map) {
  std::vector<std::pair<std::string, std::string>> input_types_list(onnx_func_proto_.input_size());
  std::vector<std::pair<std::string, std::string>> output_types_list(onnx_func_proto_.output_size());
  std::unordered_map<std::string, std::vector<std::string>> type_constraint_map;
  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto_AttributeType> attribute_type_map;

  // Create an all permissive list of data types. This will be used in case of model local functions
  // when we cannot infer the type constraints from function proto body
  std::unordered_set<std::string> all_types;
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_types_with_bfloat().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_types_with_bfloat().cend());
  all_types.insert(ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cbegin(),
                   ONNX_NAMESPACE::OpSchema::all_tensor_sequence_types().cend());

  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  std::unordered_map<std::string, int> opset_imports;
  for (auto& relied_opset : onnx_func_proto_.opset_import()) {
    opset_imports[relied_opset.domain()] = static_cast<int>(relied_opset.version());
  }
  for (auto& node : onnx_func_proto_.node()) {
    int domain_version = GetVersionForDomain(node.domain(), opset_imports);
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
          if (node_op_schema) {
            for (auto s : node_op_schema->inputs().at(i).GetTypes()) {
              type_constraint_map[type_str].emplace_back(*s);
            }
          } else {
            for (const auto& s : all_types) {
              type_constraint_map[type_str].emplace_back(s);
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
          if (node_op_schema) {
            for (auto data_type : node_op_schema->outputs().at(i).GetTypes()) {
              type_constraint_map[type_str].emplace_back(*data_type);
            }
          } else {
            for (const auto& data_type : all_types) {
              type_constraint_map[type_str].emplace_back(data_type);
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
    op_schema_->Input(i, input.first, "", input.second);
    ++i;
  }
  i = 0;
  for (auto& output : output_types_list) {
    op_schema_->Output(i, output.first, "", output.second);
    ++i;
  }

  for (auto& tc : type_constraint_map) {
    op_schema_->TypeConstraint(tc.first, tc.second, "");
  }

  for (auto& attribute_name : onnx_func_proto_.attribute()) {
    if (attribute_type_map.count(attribute_name))
      op_schema_->Attr(attribute_name, "", attribute_type_map[attribute_name], false);
  }
}

/** Utility function to initialize function body for nested model local function
  @param graph Graph in which this node belongs too. For nested functions, graph is the parent function body graph
  @param node_index index of the node in graph
  @param onnx_function_proto FunctionProto for the function
  @param in_model_function_protos Model local functions. These are schema less functions which are defined in the ModelProto of the main/parent model.
  @param function_container graph level function container which will own the initialized function body
  @param logger instance of Logger
  @param is_nested_function True if this is a nested function. For nested functions graph resolved is delayed until parent function body is fully initialized.
*/
static void InitNestedModelLocalFunction(onnxruntime::Graph& graph,
                                  const onnxruntime::NodeIndex& node_index,
                                  ONNX_NAMESPACE::FunctionProto& onnx_function_proto,
                                  const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& in_model_function_protos,
                                  std::vector<std::unique_ptr<onnxruntime::Function>>& function_container,
                                  const logging::Logger& logger,
                                  bool is_nested_function) {
  ORT_TRY {
    auto func_ptr = std::make_unique<onnxruntime::FunctionImpl>(graph, node_index, onnx_function_proto,
                                                                in_model_function_protos, function_container,
                                                                logger, is_nested_function);
    function_container.emplace_back(std::move(func_ptr));
    auto* node_in_graph = graph.GetNode(node_index);
    node_in_graph->SetFunctionBody(*function_container.back());
  }
  ORT_CATCH(const std::exception& e) {
    LOGS(logger, WARNING) << "Function body initialization failed for Function '"
                          << onnx_function_proto.name()
#ifndef ORT_NO_EXCEPTIONS
                          << "'. Error message " << e.what()
#endif //ORT_NO_EXCEPTIONS
                          << ". Execution will fail if ORT does not have a specialized kernel for this op";
    // Return without using this function op's expansion. No need to fail just yet.
    // If ORT has a specialized kernel for this op then execution will proceed
    return;
  }
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
                                                  const std::unordered_map<std::string, int>& input_name_idx_map,
                                                  const std::unordered_map<std::string, int>& output_name_idx_map) {
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

static std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
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

// Creates domain to version map for onnx function
static std::unordered_map<std::string, int> GetFunctionOpsetImports(const ONNX_NAMESPACE::FunctionProto& func_proto, const std::unordered_map<std::string, int>& graph_imports) {
  std::unordered_map<std::string, int> function_opset_imports{graph_imports};
  for (const auto& opset_import : func_proto.opset_import()) {
    // If graph imports does not contain opset_import then insert it otherwise the one in graph imports overrides.
    // If the opset imports are not compatible then this will be caught during function body inline.
    function_opset_imports.insert({opset_import.domain(), static_cast<int>(opset_import.version())});
  }
  return function_opset_imports;
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const IndexedSubGraph& nodes_to_fuse,
                           const logging::Logger& logger)
    : parent_graph_(&graph),
      body_("fused_function_subgraph", false, onnxruntime::ModelMetaData(),
            graph.ModelPath().ToPathString(),
            IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}),
            graph.DomainToVersionMap(), {} , logger) {
  auto& function_body_graph = body_.MainGraph();

  auto* meta_def = nodes_to_fuse.GetMetaDef();
  op_schema_ = CreateSchema(graph, nodes_to_fuse);

  int i = 0;
  std::vector<const NodeArg*> function_body_graph_inputs;
  function_body_graph_inputs.resize(meta_def->inputs.size());
  for (auto& input : meta_def->inputs) {
    auto input_arg = parent_graph_->GetNodeArg(input);
    auto& function_body_graph_input_arg = function_body_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
    function_body_graph_inputs[i] = &function_body_graph_input_arg;
    ++i;
  }

  i = 0;
  std::vector<const NodeArg*> function_body_graph_outputs;
  function_body_graph_outputs.resize(meta_def->outputs.size());
  for (auto& output : meta_def->outputs) {
    auto output_arg = parent_graph_->GetNodeArg(output);
    auto& function_body_graph_output_arg = function_body_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
    function_body_graph_outputs[i] = &function_body_graph_output_arg;
    ++i;
  }

  function_body_graph.SetInputs(function_body_graph_inputs);
  function_body_graph.SetOutputs(function_body_graph_outputs);

  //Add node and node args
  //TODO: for better performance, we could try to transfer the nodes in parent graph to sub-graph directly,
  //instead of create new nodes.
  for (auto& node_index : nodes_to_fuse.nodes) {
    auto node = parent_graph_->GetNode(node_index);
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;
    for (auto input : node->InputDefs()) {
      auto& n_input = function_body_graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node->OutputDefs()) {
      auto& n_output = function_body_graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    function_body_graph.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
  }

  for (const auto& input : meta_def->inputs) {
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
    if (graph.GetInitializedTensor(input, initializer)) {
      // meta_def->inputs could have duplicates so make sure we only add once
      const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
      if (!function_body_graph.GetInitializedTensor(input, subgraph_initializer)) {
        function_body_graph.AddInitializedTensor(*initializer);
      }
    }
  }

  //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
  auto status = function_body_graph.Resolve();
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
}

FunctionImpl::FunctionImpl(onnxruntime::Graph& graph,
                           const onnxruntime::NodeIndex& node_index,
                           const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                           const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
                           std::vector<std::unique_ptr<onnxruntime::Function>>& function_container,
                           const logging::Logger& logger,
                           bool is_nested_function)
    : parent_graph_(&graph),
      body_(onnx_func_proto.name(), false, onnxruntime::ModelMetaData(),
            graph.ModelPath().ToPathString(), IOnnxRuntimeOpSchemaRegistryList(),
            onnx_func_proto.opset_import_size() != 0 ? GetFunctionOpsetImports(onnx_func_proto, graph.DomainToVersionMap()) : graph.DomainToVersionMap(),
            {}, logger),
      onnx_func_proto_(onnx_func_proto) {
  // Make a copy of the FunctionProto.
  // All FunctionBody ops with the same op type seem to share the same FunctionProto struct within a model.
  // Hence, we make a copy prior to generating the graph representation of the function,
  // as we might make some modifications to the FunctionProto along the way

  const auto* node_in_parent_graph = parent_graph_->GetNode(node_index);

  // For schema defined functions get the version from the node in parent graph.
  // For the functions which do not have schema defined (model local functions)
  // get the since version from the version in opset imports using the domain.
  auto since_version = node_in_parent_graph->SinceVersion() == -1
                           ? GetVersionForDomain(node_in_parent_graph->Domain(), body_.MainGraph().DomainToVersionMap())
                           : node_in_parent_graph->SinceVersion();
  op_schema_ = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema_->SetName(onnx_func_proto_.name());
  op_schema_->SetDomain(node_in_parent_graph->Domain());
  op_schema_->SetDoc(onnx_func_proto_.doc_string());
  op_schema_->SinceVersion(static_cast<ONNX_NAMESPACE::OperatorSetVersion>(since_version));
  std::unordered_map<std::string, int> input_name_idx_map;
  std::unordered_map<std::string, int> output_name_idx_map;
  std::unordered_map<std::string, std::string> internal_input_output_updates;

  auto& function_body_graph = body_.MainGraph();

  for (int i = 0; i < onnx_func_proto_.input_size(); ++i) {
    input_name_idx_map[onnx_func_proto_.input().Get(i)] = i;
  }
  for (int i = 0; i < onnx_func_proto_.output_size(); ++i) {
    output_name_idx_map[onnx_func_proto_.output().Get(i)] = i;
  }

  auto cached_op_schema = node_in_parent_graph->Op();
  if (!cached_op_schema) {
    // Infer a op_schema for stand-alone functions.
    IOTypeConstraintHelper(onnx_func_proto_, op_schema_, input_name_idx_map, output_name_idx_map);
  } else {
    auto type_constraint_params = cached_op_schema->typeConstraintParams();
    for (auto& type_constraint_param : type_constraint_params) {
      op_schema_->TypeConstraint(type_constraint_param.type_param_str,
                                 type_constraint_param.allowed_type_strs,
                                 type_constraint_param.description);
    }
    int i = 0;
    for (auto& input : cached_op_schema->inputs()) {
      op_schema_->Input(i, input.GetName(), input.GetDescription(), input.GetTypeStr(), input.GetOption(),
                        input.GetIsHomogeneous(), input.GetMinArity());
      ++i;
    }
    i = 0;
    for (auto& output : cached_op_schema->outputs()) {
      op_schema_->Output(i, output.GetName(), output.GetDescription(), output.GetTypeStr(), output.GetOption(),
                         output.GetIsHomogeneous(), output.GetMinArity());
      ++i;
    }
    for (auto& attribute : cached_op_schema->attributes()) {
      op_schema_->Attr(attribute.second);
    }
  }

  if (!cached_op_schema || !cached_op_schema->has_type_and_shape_inference_function()) {
    op_schema_->TypeAndShapeInferenceFunction(
        [this, &model_local_functions](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
          ONNX_NAMESPACE::ShapeInferenceOptions options {true, 1, false};
          InferShapeForFunctionNode(ctx, onnx_func_proto_, body_.MainGraph().DomainToVersionMap(), options, schema_registry, model_local_functions, function_utils::GetFunctionIdentifier);
        });
  } else {
    op_schema_->TypeAndShapeInferenceFunction(cached_op_schema->GetTypeAndShapeInferenceFunction());
  }

  op_schema_->Finalize();
  //construct body
  std::vector<const NodeArg*> graph_inputs(node_in_parent_graph->InputDefs().size(), nullptr),
      graph_outputs(node_in_parent_graph->OutputDefs().size(), nullptr);

  // Add node and node args into subgraph
  // The subgraph preserved the input/output tensor names
  // in the parent graph for later inlining purpose
  const auto& attr_map = node_in_parent_graph->GetAttributes();

  ONNX_NAMESPACE::NodeProto function_op_node_proto;  // NodeProto pertaining to the op with a FunctionBody
  node_in_parent_graph->ToProto(function_op_node_proto);
  std::unordered_set<std::string> node_input_outputs;

  for (const auto* input_def : node_in_parent_graph->InputDefs()) {
    if (input_def->Exists()) {
      node_input_outputs.insert(input_def->Name());
    }
  }

  for (const auto* output_def : node_in_parent_graph->OutputDefs()) {
    if (output_def->Exists()) {
      node_input_outputs.insert(output_def->Name());
    }
  }

  ONNX_NAMESPACE::TypeProto tensor_int32;  // dummy type used for unused formal parameters
  tensor_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // iterate over each node in the FunctionProto and fix inputs/outputs
  for (auto node = onnx_func_proto_.mutable_node()->begin(); node != onnx_func_proto_.mutable_node()->end(); ++node) {
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;
    std::string uniq_identifier = (*node).name();
    if (!utils::HasName(*node)) {
      std::stringstream ss;
      ss << static_cast<const void*>(&(*node));
      uniq_identifier = ss.str();
    }

    for (int idx = 0; idx < (*node).input_size(); ++idx) {
      const std::string& tensor_name = (*node).input().Get(idx);
      auto iter = input_name_idx_map.find(tensor_name);
      if (iter != input_name_idx_map.end()) {
        // If input is part of function inputs, preserve NodeArg and input/output names
        const std::string& actual_parameter_name = function_op_node_proto.input().Get(iter->second);
        if (!actual_parameter_name.empty()) {
          const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(actual_parameter_name);
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
          const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(actual_parameter_name);
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
      auto iter = output_name_idx_map.find(tensor_name);
      if (iter != output_name_idx_map.end()) {
        // Preserving NodeArg and input/output names
        const std::string& actual_parameter_name = function_op_node_proto.output().Get(iter->second);
        if (!actual_parameter_name.empty()) {
          const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(actual_parameter_name);
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
        auto tensor_name = onnx_func_proto_.input(i) + "_dummy";
        auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
        graph_inputs[i] = &unused_formal_param;
      }
    }

    for (unsigned i = 0; i < graph_outputs.size(); ++i) {
      if (graph_outputs[i] == nullptr) {
        auto tensor_name = onnx_func_proto_.output(i) + "_dummy";
        auto& unused_formal_param = function_body_graph.GetOrCreateNodeArg(tensor_name, &tensor_int32);
        graph_outputs[i] = &unused_formal_param;
      }
    }

    onnxruntime::NodeAttributes new_attr_map;
    for (auto node_attr = (*node).mutable_attribute()->begin();
         node_attr != (*node).mutable_attribute()->end(); ++node_attr) {
      // If this node contains subgraphs, the node inputs/outputs within them needs to be fixed as well
      if ((*node_attr).has_g()) {
        UpdateSubgraphsWithinFunctionBody(*(*node_attr).mutable_g(),
                                              *parent_graph_, function_op_node_proto,
                                              input_name_idx_map, output_name_idx_map);
      }

      if (!(*node_attr).ref_attr_name().empty()) {
        auto entry = attr_map.find((*node_attr).ref_attr_name());
        if (entry != attr_map.cend()) {
          onnx::AttributeProto attr_copy = entry->second;
          attr_copy.set_name(node_attr->name());
          new_attr_map[(*node_attr).name()] = attr_copy;
        }
      } else {
        new_attr_map[(*node_attr).name()] = *node_attr;
      }
    }
    function_body_graph.AddNode(uniq_identifier, (*node).op_type(),
                                (*node).doc_string(), inputs, outputs, &new_attr_map, (*node).domain());
  }

  function_body_graph.SetInputs(graph_inputs);
  function_body_graph.SetOutputs(graph_outputs);

  // Nested model local functions need to be initialized before Graph::Resolve() can be called for the function body.
  // Parse the graph and initialize functions for nodes which reference model local functions.
  // Only parse the graph if the model contains model local functions.
  // Once all model local functions within function body are initialized, Graph Resolve for parent function body is called
  // During graph resolve for parent function, graph resolve for every nested model local function is called too...
  // Such a top down approach is required to successfully carry out type inference for schema less functions.
  // Schema defined functions are treated a bit different from model local aka schema less functions. These are initialized
  // during graph resolve of parent functions.
  if (model_local_functions.size() > 0) {
    for (auto node = function_body_graph.Nodes().begin(); node != function_body_graph.Nodes().end(); ++node) {
      // Init nested functions
      std::string func_identifier = function_utils::GetFunctionIdentifier(node->Domain(), node->OpType());
      auto iter = model_local_functions.find(func_identifier);
      if (iter == model_local_functions.end()) {
        continue;
      }

      // This node has a model local function proto.
      auto onnx_function_proto = *(iter->second);
      InitNestedModelLocalFunction(function_body_graph, node->Index(), onnx_function_proto, model_local_functions, function_container, logger, true);
    }
  }

  // Graph resolve should be called on the parent functions only. Skip resolve if this is a nested function.
  // Nested function bodies will be resolved along with parent function body as we set traverse_function_body to true.
  // This is only applicable for model local functions which are schema less.
  if (!is_nested_function) {
    onnxruntime::Graph::ResolveOptions options;
    options.traverse_function_body = true;
    auto status = function_body_graph.Resolve(options);

    ORT_ENFORCE(status.IsOK(), "Resolve subgraph failed:", status.ErrorMessage());

    ORT_ENFORCE(node_in_parent_graph->InputDefs().size() == function_body_graph.GetInputsIncludingInitializers().size(),
                "Node " + node_in_parent_graph->Name() + "'s number of inputs is different from function body graph's number of input.");

    ORT_ENFORCE(node_in_parent_graph->OutputDefs().size() == function_body_graph.GetOutputs().size(),
                "Node ", node_in_parent_graph->Name(), "'s number of outputs is different from function body graph's number of outputs.");
  }

}  // namespace onnxruntime

FunctionImpl::~FunctionImpl() = default;

const ONNX_NAMESPACE::OpSchema& FunctionImpl::OpSchema() const {
  return *op_schema_;
}

const onnxruntime::Graph& FunctionImpl::Body() const {
  return body_.MainGraph();
}

onnxruntime::Graph& FunctionImpl::MutableBody() {
    return body_.MainGraph();
}

ViewerFunctionImpl::ViewerFunctionImpl(const onnxruntime::Graph& graph,
                                       const IndexedSubGraph& nodes_to_fuse,
                                       const logging::Logger& /*logger*/) {
  op_schema_ = CreateSchema(graph, nodes_to_fuse);
}

ViewerFunctionImpl::~ViewerFunctionImpl() = default;

std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       const IndexedSubGraph& nodes_to_fuse,
                                       const logging::Logger& logger) {
  return std::make_unique<FunctionImpl>(graph, nodes_to_fuse, logger);
}
}  // namespace onnxruntime
