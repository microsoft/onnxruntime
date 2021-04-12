// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxruntime {
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
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  for (auto& node : onnx_func_proto_.node()) {
    const auto node_op_schema =
        schema_registry->GetSchema(node.op_type(), static_cast<int>(onnx_func_proto_.since_version()), node.domain());
    for (int i = 0; i < node.input_size(); ++i) {
      auto& in_name = node.input().Get(i);
      auto iter = input_name_idx_map.find(in_name);
      if (iter != input_name_idx_map.end()) {
        int idx = iter->second;
        const auto& p = node_op_schema->inputs().at(i);
        std::string type_str = p.GetTypeStr() + "in" + std::to_string(i);
        input_types_list[idx] = std::make_pair(in_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          for (auto s : p.GetTypes()) {
            type_constraint_map[type_str].emplace_back(*s);
          }
        }
      }
    }
    for (int i = 0; i < node.output_size(); ++i) {
      auto& out_name = node.output().Get(i);
      auto iter = output_name_idx_map.find(out_name);
      if (iter != output_name_idx_map.end()) {
        int idx = iter->second;
        const auto& p = node_op_schema->outputs().at(i);
        std::string type_str = p.GetTypeStr() + "out" + std::to_string(i);
        output_types_list[idx] = std::make_pair(out_name, type_str);
        if (!type_constraint_map.count(type_str)) {
          for (auto s : p.GetTypes()) {
            type_constraint_map[type_str].emplace_back(*s);
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
static void update_subgraphs_within_function_body(ONNX_NAMESPACE::GraphProto& subgraph_proto,
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
        update_subgraphs_within_function_body(*(*subgraph_node_attr).mutable_g(),
                                              parent_graph, function_node_in_parent_graph,
                                              input_name_idx_map, output_name_idx_map);
      }
    }
  }
}

static std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
                                                              const IndexedSubGraph& nodes_to_fuse) {
  const auto* meta_def = nodes_to_fuse.GetMetaDef();
  auto op_schema = onnxruntime::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema->SetName(meta_def->name);
  op_schema->SetDomain(meta_def->domain);
  op_schema->SetDoc(meta_def->doc_string);
  op_schema->SinceVersion(meta_def->since_version);
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

// Creates domain to version map for onnx function by merging graph level opset imports with opset imports from 
// funtion proto
static std::unordered_map<std::string, int> CreateOpsetImportsForFunction(const ONNX_NAMESPACE::FunctionProto& func_proto,
                                                                          const std::unordered_map<std::string, int>& graph_opset_imports) {
  // function inherits all graph level opset imports
  std::unordered_map<std::string, int> function_opset_imports{graph_opset_imports};
  // merge with opset imports in function proto
  for (const auto& opset_import : func_proto.opset_import()) {
    auto opset_version = static_cast<int>(opset_import.version());
    auto result = function_opset_imports.insert({opset_import.domain(), opset_version});
    ORT_ENFORCE((result.first->second == opset_version),
                "ONNX model does not support multiple opset versions for a domain. Model imports opset version ",
                result.first->second, " for domain ", result.first->first, " and function is trying to import opset version ",
                opset_version, " for the same domain");
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
            graph.DomainToVersionMap(), {}, logger) {
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

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const onnxruntime::NodeIndex& node_index,
                           const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                           const logging::Logger& logger)
    : parent_graph_(&graph),
      body_(onnx_func_proto.name(), false, onnxruntime::ModelMetaData(),
            graph.ModelPath().ToPathString(), IOnnxRuntimeOpSchemaRegistryList(),
            CreateOpsetImportsForFunction(onnx_func_proto, graph.DomainToVersionMap()),
            {}, logger),
      onnx_func_proto_(onnx_func_proto) {
  // Make a copy of the FunctionProto.
  // All FunctionBody ops with the same op type seem to share the same FunctionProto struct within a model.
  // Hence, we make a copy prior to generating the graph representation of the function,
  // as we might make some modifications to the FunctionProto along the way

  const auto* node_in_parent_graph = parent_graph_->GetNode(node_index);
  op_schema_ = onnxruntime::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema_->SetName(onnx_func_proto_.name());
  op_schema_->SetDomain(node_in_parent_graph->Domain());
  op_schema_->SetDoc(onnx_func_proto_.doc_string());
  op_schema_->SinceVersion(static_cast<ONNX_NAMESPACE::OperatorSetVersion>(onnx_func_proto_.since_version()));
  std::unordered_map<std::string, int> input_name_idx_map;
  std::unordered_map<std::string, int> output_name_idx_map;
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
        [this](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
          ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(&onnx_func_proto_, body_.MainGraph().DomainToVersionMap(), schema_registry, ctx);
        });
  } else {
    op_schema_->TypeAndShapeInferenceFunction(cached_op_schema->GetTypeAndShapeInferenceFunction());
  }

  op_schema_->Finalize();
  //construct body
  auto& function_body_graph = body_.MainGraph();
  std::vector<const NodeArg*> graph_inputs(node_in_parent_graph->InputDefs().size(), nullptr),
      graph_outputs(node_in_parent_graph->OutputDefs().size(), nullptr);

  // Add node and node args into subgraph
  // The subgraph preserved the input/output tensor names
  // in the parent graph for later inlining purpose
  const auto& attr_map = node_in_parent_graph->GetAttributes();

  ONNX_NAMESPACE::NodeProto function_op_node_proto;  // NodeProto pertaining to the op with a FunctionBody
  node_in_parent_graph->ToProto(function_op_node_proto);

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
        // Preserving NodeArg and input/output names
        const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(function_op_node_proto.input()
                                                                             .Get(iter->second));
        auto& n_input = function_body_graph.GetOrCreateNodeArg(
            function_op_node_proto.input().Get(iter->second), node_arg->TypeAsProto());
        inputs.push_back(&n_input);
        graph_inputs[iter->second] = &n_input;
      } else {
        auto& n_input = function_body_graph.GetOrCreateNodeArg(
            tensor_name + "_" + std::to_string(node_index), nullptr);
        inputs.push_back(&n_input);
      }
    }
    for (int idx = 0; idx < (*node).output_size(); ++idx) {
      std::string tensor_name = (*node).output().Get(idx);
      auto iter = output_name_idx_map.find(tensor_name);
      if (iter != output_name_idx_map.end()) {
        // Preserving NodeArg and input/output names
        const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(function_op_node_proto.output()
                                                                             .Get(iter->second));
        auto& n_output = function_body_graph.GetOrCreateNodeArg(
            function_op_node_proto.output().Get(iter->second), node_arg->TypeAsProto());
        outputs.push_back(&n_output);
        graph_outputs[iter->second] = &n_output;
      } else {
        auto& n_output = function_body_graph.GetOrCreateNodeArg(
            tensor_name + "_" + std::to_string(node_index), nullptr);
        outputs.push_back(&n_output);
      }
    }

    onnxruntime::NodeAttributes new_attr_map;
    for (auto node_attr = (*node).mutable_attribute()->begin();
         node_attr != (*node).mutable_attribute()->end(); ++node_attr) {
      // If this node contains subgraphs, the node inputs/outputs within them needs to be fixed as well
      if ((*node_attr).has_g()) {
        update_subgraphs_within_function_body(*(*node_attr).mutable_g(),
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
    function_body_graph.AddNode(uniq_identifier + "_" + std::to_string(node_index), (*node).op_type(),
                                (*node).doc_string(), inputs, outputs, &new_attr_map, (*node).domain());
  }

  function_body_graph.SetInputs(graph_inputs);
  function_body_graph.SetOutputs(graph_outputs);
  auto status = function_body_graph.Resolve();

  ORT_ENFORCE(status.IsOK(), "Resolve subgraph failed:", status.ErrorMessage());
}  // namespace onnxruntime

FunctionImpl::~FunctionImpl() = default;

const ONNX_NAMESPACE::OpSchema& FunctionImpl::OpSchema() const {
  return *op_schema_;
}

const onnxruntime::Graph& FunctionImpl::Body() const {
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
  return onnxruntime::make_unique<FunctionImpl>(graph, nodes_to_fuse, logger);
}
}  // namespace onnxruntime
