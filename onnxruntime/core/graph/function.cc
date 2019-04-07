// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxruntime {
// Auto inferred and generate an opschema for stand-alone functions
// TODO: revisit to see if we can eliminate typeconstraint step
void IOTypeConstraintHelper(const ONNX_NAMESPACE::FunctionProto* onnx_func_proto_,
                          std::unique_ptr<ONNX_NAMESPACE::OpSchema>& op_schema_,
                          const std::unordered_map<std::string, int>& input_name_idx_map,
                          const std::unordered_map<std::string, int>& output_name_idx_map) {
  std::vector<std::pair<std::string, std::string>> input_types_list(onnx_func_proto_->input_size());
  std::vector<std::pair<std::string, std::string>> output_types_list(onnx_func_proto_->output_size());
  std::unordered_map<std::string, std::vector<std::string>> type_constraint_map;
  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto_AttributeType> attribute_type_map;
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  for (auto& node : onnx_func_proto_->node()) {
    const auto node_op_schema = schema_registry->GetSchema(node.op_type(), (int)onnx_func_proto_->since_version(), node.domain());
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
      if (attr.has_ref_attr_name() && attr.has_type())
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

  for (auto& attribute_name : onnx_func_proto_->attribute()) {
    if (attribute_type_map.count(attribute_name))
      op_schema_->Attr(attribute_name, "", attribute_type_map[attribute_name], false);
  }
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           std::unique_ptr<IndexedSubGraph> customized_func)
    : parent_graph_(&graph), onnx_func_proto_{nullptr} {
  customized_func_body_ = std::move(customized_func);
  auto meta_def = customized_func_body_->GetMetaDef();
  op_schema_ = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema_->SetName(meta_def->name);
  op_schema_->SetDomain(meta_def->domain);
  op_schema_->SetDoc(meta_def->doc_string);
  op_schema_->SinceVersion(meta_def->since_version);
  int i = 0;
  for (auto& input : meta_def->inputs) {
    auto input_type = parent_graph_->GetNodeArg(input)->Type();
    op_schema_->Input(i, input, "", *input_type);
    ++i;
  }
  i = 0;
  for (auto& output : meta_def->outputs) {
    auto output_type = parent_graph_->GetNodeArg(output)->Type();
    op_schema_->Output(i, output, "", *output_type);
    ++i;
  }
  op_schema_->Finalize();
  //construct body
  body_ = std::make_unique<onnxruntime::Model>("fused_function_subgraph", false, onnxruntime::ModelMetaData(),
                                               IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}),
                                               graph.DomainToVersionMap());

  auto& sub_graph = body_->MainGraph();
  //Add node and node args
  //TODO: for better performance, we could try to transfer the nodes in parent graph to sub-graph directly,
  //instead of create new nodes.
  for (auto& node_index : customized_func_body_->nodes) {
    auto node = parent_graph_->GetNode(node_index);
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node->InputDefs()) {
      auto& n_input = sub_graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node->OutputDefs()) {
      auto& n_output = sub_graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    sub_graph.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
  }

  for (auto input : meta_def->inputs) {
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
    if (graph.GetInitializedTensor(input, initializer)) {
      sub_graph.AddInitializedTensor(*initializer);
    }
  }

  //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
  ORT_ENFORCE(sub_graph.Resolve().IsOK());
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const onnxruntime::NodeIndex& node_index,
                           const ONNX_NAMESPACE::FunctionProto* onnx_func_proto)
    : parent_graph_(&graph) {
  onnx_func_proto_ = onnx_func_proto;
  auto node_in_parent_graph = parent_graph_->GetNode(node_index);
  op_schema_ = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema_->SetName(onnx_func_proto_->name());
  op_schema_->SetDomain(onnx_func_proto_->node().Get(0).domain());
  op_schema_->SetDoc(onnx_func_proto_->doc_string());
  op_schema_->SinceVersion((ONNX_NAMESPACE::OperatorSetVersion)onnx_func_proto_->since_version());
  std::unordered_map<std::string, int> input_name_idx_map;
  std::unordered_map<std::string, int> output_name_idx_map;
  for (int i = 0; i < onnx_func_proto_->input_size(); ++i) {
    input_name_idx_map[onnx_func_proto_->input().Get(i)] = i;
  }
  for (int i = 0; i < onnx_func_proto_->output_size(); ++i) {
    output_name_idx_map[onnx_func_proto_->output().Get(i)] = i;
  }

  auto cached_op_schema = node_in_parent_graph->Op();
  if (!cached_op_schema) {
    // Infer a op_schema for stand-alone functions.
    IOTypeConstraintHelper(onnx_func_proto_, this->op_schema_, input_name_idx_map, output_name_idx_map);
  } else {
    auto type_constraint_params = cached_op_schema->typeConstraintParams();
    for (auto& type_constraint_param : type_constraint_params) {
      op_schema_->TypeConstraint(
        type_constraint_param.type_param_str, 
        type_constraint_param.allowed_type_strs, 
        type_constraint_param.description);
    }
    int i = 0;
    for (auto& input : cached_op_schema->inputs()) {
      op_schema_->Input(i, input.GetName(), input.GetDescription(), input.GetTypeStr());
      ++i;
    }
    i = 0;
    for (auto& output : cached_op_schema->outputs()) {
      op_schema_->Output(i, output.GetName(), output.GetDescription(), output.GetTypeStr());
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
      const ONNX_NAMESPACE::FunctionProto* func_ptr = this->GetFuncProto();
      if (nullptr != func_ptr) {
        ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(func_ptr, schema_registry, ctx);
      }
    });
  } else {
    op_schema_->TypeAndShapeInferenceFunction(cached_op_schema->GetTypeAndShapeInferenceFunction());
  }

  op_schema_->Finalize();
  //construct body
  std::unordered_map<std::string, int> domain_to_version;
  //TODO: set correct domain and version
  domain_to_version[onnxruntime::kOnnxDomain] = (int)onnx_func_proto_->since_version();
  body_ = std::make_unique<onnxruntime::Model>(onnx_func_proto_->name(), false, onnxruntime::ModelMetaData(),
                                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
  auto& sub_graph = body_->MainGraph();
  // Add node and node args into subgraph
  // The subgraph preserved the input/output tensor names
  // in the parent graph for later inlining purpose
  auto attr_map = node_in_parent_graph->GetAttributes();
  for (auto& node : onnx_func_proto_->node()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    std::string uniq_identifier = node.name();
    if (!node.has_name()) {
      std::stringstream ss;
      ss << static_cast<const void*>(&node);
      uniq_identifier = ss.str();
    }

    for (int idx = 0; idx < node.input_size(); ++idx) {
      std::string tensor_name = node.input().Get(idx);
      auto iter = input_name_idx_map.find(tensor_name);
      if (iter != input_name_idx_map.end()) {
        // Preserving NodeArg and input/output names
        ONNX_NAMESPACE::NodeProto temp_node_proto;
        node_in_parent_graph->ToProto(temp_node_proto);
        const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(temp_node_proto.input().Get(input_name_idx_map[tensor_name]));
        auto& n_input = sub_graph.GetOrCreateNodeArg(
            temp_node_proto.input().Get(iter->second), node_arg->TypeAsProto());
        inputs.push_back(&n_input);
      } else {
        auto& n_input = sub_graph.GetOrCreateNodeArg(
            tensor_name + "_" + std::to_string(node_index), nullptr);
        inputs.push_back(&n_input);
      }
    }
    for (int idx = 0; idx < node.output_size(); ++idx) {
      std::string tensor_name = node.output().Get(idx);
      auto iter = output_name_idx_map.find(tensor_name);
      if (iter != output_name_idx_map.end()) {
        // Preserving NodeArg and input/output names
        ONNX_NAMESPACE::NodeProto temp_node_proto;
        node_in_parent_graph->ToProto(temp_node_proto);
        const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(temp_node_proto.output().Get(output_name_idx_map[tensor_name]));
        auto& n_output = sub_graph.GetOrCreateNodeArg(
            temp_node_proto.output().Get(iter->second), node_arg->TypeAsProto());
        outputs.push_back(&n_output);
      } else {
        auto& n_output = sub_graph.GetOrCreateNodeArg(
            tensor_name + "_" + std::to_string(node_index), nullptr);
        outputs.push_back(&n_output);
      }
    }

    onnxruntime::NodeAttributes new_attr_map;
    for (auto& attr : node.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          new_attr_map[attr.name()] = attr_map[attr.ref_attr_name()];
        }
      } else {
        new_attr_map[attr.name()] = attr;
      }
    }
    sub_graph.AddNode(uniq_identifier + "_" + std::to_string(node_index), node.op_type(), node.doc_string(), inputs, outputs, &new_attr_map, node.domain());
  }
  auto status = sub_graph.Resolve();
  ORT_ENFORCE(status.IsOK(), "Resolve subgraph failed:", status.ErrorMessage());
}

FunctionImpl::~FunctionImpl() = default;

const ONNX_NAMESPACE::OpSchema& FunctionImpl::OpSchema() const {
  return *op_schema_;
}

const onnxruntime::Graph& FunctionImpl::Body() const {
  return body_->MainGraph();
}

const IndexedSubGraph& FunctionImpl::GetIndexedSubGraph() const {
  return *customized_func_body_;
}

const ONNX_NAMESPACE::FunctionProto* FunctionImpl::GetFuncProto() const {
  return onnx_func_proto_;
}

std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       std::unique_ptr<IndexedSubGraph> customized_func) {
  return std::make_unique<FunctionImpl>(graph, std::move(customized_func));
}
}  // namespace onnxruntime
