// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/function_impl.h"
#include "core/graph/graph.h"
#include "core/graph/function_container.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxruntime {
void TypeConstraintHelper(const ONNX_NAMESPACE::FunctionProto* onnx_func_proto_,
                          std::unique_ptr<ONNX_NAMESPACE::OpSchema>& op_schema_,
                          /*out*/
                          std::unordered_map<std::string, int>& input_name_idx_map,
                          std::unordered_map<std::string, int>& output_name_idx_map) {
  std::vector<std::pair<std::string, std::string>> input_types_list(onnx_func_proto_->input_size());
  std::vector<std::pair<std::string, std::string>> output_types_list(onnx_func_proto_->output_size());
  std::unordered_map<std::string, std::vector<std::string>> type_constraint_map;
  for (int i = 0; i < onnx_func_proto_->input_size(); ++i) {
    input_name_idx_map[onnx_func_proto_->input().Get(i)] = i;
  }
  for (int i = 0; i < onnx_func_proto_->output_size(); ++i) {
    output_name_idx_map[onnx_func_proto_->output().Get(i)] = i;
  }
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  for (auto& node : onnx_func_proto_->node()) {
    const auto node_op_schema = schema_registry->GetSchema(node.op_type(), (int)onnx_func_proto_->since_version(), node.domain());
    for (int i = 0; i < node.input_size(); ++i) {
      auto& in_name = node.input().Get(i);
      if (input_name_idx_map.count(in_name)) {
        int idx = input_name_idx_map[in_name];
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
      if (output_name_idx_map.count(out_name)) {
        int idx = output_name_idx_map[out_name];
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
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           std::unique_ptr<IndexedSubGraph> customized_func)
    : parent_graph_(&graph) {
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
                                               IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}), graph.DomainToVersionMap());

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
  //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
  ONNXRUNTIME_ENFORCE(sub_graph.Resolve().IsOK());
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const onnxruntime::NodeIndex& node_index,
                           const ONNX_NAMESPACE::FunctionProto* onnx_func_proto)
    : parent_graph_(&graph) {
  onnx_func_proto_ = onnx_func_proto;
  auto node_in_parent_graph = parent_graph_->GetNode(node_index);
  op_schema_ = std::make_unique<onnx::OpSchema>();
  op_schema_->SetName(onnx_func_proto_->name());
  op_schema_->SetDomain(onnx_func_proto_->node().Get(0).domain());
  op_schema_->SetDoc(onnx_func_proto_->doc_string());
  op_schema_->SinceVersion((ONNX_NAMESPACE::OperatorSetVersion)onnx_func_proto_->since_version());
  std::unordered_map<std::string, int> input_name_idx_map;
  std::unordered_map<std::string, int> output_name_idx_map;
  TypeConstraintHelper(onnx_func_proto_, this->op_schema_, input_name_idx_map, output_name_idx_map);

  op_schema_->TypeAndShapeInferenceFunction(
      [this](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        const ONNX_NAMESPACE::FunctionProto* func_ptr = this->GetFuncProto();
        if (nullptr != func_ptr) {
          ONNX_NAMESPACE::shape_inference::InferShapeForFunctionNode(*func_ptr, schema_registry, ctx);
        }
      });

  op_schema_->Finalize();
  //construct body
  std::unordered_map<std::string, int> domain_to_version;
  //TODO: set correct domain and version
  domain_to_version[onnxruntime::kOnnxDomain] = (int)onnx_func_proto_->since_version();
  body_ = std::make_unique<onnxruntime::Model>(onnx_func_proto_->name(), false, onnxruntime::ModelMetaData(),
                                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
  auto& sub_graph = body_->MainGraph();
  //Add node and node args into subgraph
  auto attr_map = node_in_parent_graph->GetAttributes();
  for (auto& node : onnx_func_proto_->node()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (int idx = 0; idx < node.input_size(); ++idx) {
      std::string tensor_name = node.input().Get(idx);
      if (input_name_idx_map.count(tensor_name)) {
        ONNX_NAMESPACE::NodeProto temp_node_proto;
        node_in_parent_graph->ToProto(temp_node_proto);
        const onnxruntime::NodeArg* node_arg = parent_graph_->GetNodeArg(temp_node_proto.input().Get(input_name_idx_map[tensor_name]));
        auto& n_input = sub_graph.GetOrCreateNodeArg(
            tensor_name, node_arg->TypeAsProto());
        inputs.push_back(&n_input);
      } else {
        auto& n_input = sub_graph.GetOrCreateNodeArg(
            tensor_name, nullptr);
        inputs.push_back(&n_input);
      }
    }
    for (int idx = 0; idx < node.output_size(); ++idx) {
      std::string tensor_name = node.output().Get(idx);
      auto& n_output = sub_graph.GetOrCreateNodeArg(tensor_name, nullptr);
      outputs.push_back(&n_output);
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
    sub_graph.AddNode(node.name(), node.op_type(), node.doc_string(), inputs, outputs, &new_attr_map, node.domain());
  }
  auto status = sub_graph.Resolve();
  ONNXRUNTIME_ENFORCE(status.IsOK());
}

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
