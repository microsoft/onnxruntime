// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"

namespace onnxruntime {
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
// Construct it with fused index graph, instantiate the function directly
FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const IndexedSubGraph& nodes_to_fuse,
                           const logging::Logger& logger)
    : body_("fused_function_subgraph", false, onnxruntime::ModelMetaData(),
            graph.ModelPath().ToPathString(),
            IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}),
            graph.DomainToVersionMap(), {}, logger){
  auto& function_body_graph = body_.MainGraph();

  auto* meta_def = nodes_to_fuse.GetMetaDef();

  int i = 0;
  std::vector<const NodeArg*> function_body_graph_inputs;
  function_body_graph_inputs.resize(meta_def->inputs.size());
  for (auto& input : meta_def->inputs) {
    auto input_arg = graph.GetNodeArg(input);
    auto& function_body_graph_input_arg = function_body_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
    function_body_graph_inputs[i] = &function_body_graph_input_arg;
    ++i;
  }

  i = 0;
  std::vector<const NodeArg*> function_body_graph_outputs;
  function_body_graph_outputs.resize(meta_def->outputs.size());
  for (auto& output : meta_def->outputs) {
    auto output_arg = graph.GetNodeArg(output);
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
    auto node = graph.GetNode(node_index);
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

  for (const auto& constant_initializer : meta_def->constant_initializers) {
    const ONNX_NAMESPACE::TensorProto* initializer = graph.GetConstantInitializer(constant_initializer, true);
    ORT_ENFORCE(initializer != nullptr, "Initializer " + constant_initializer + " is not found or is not constant initializer.");
    // meta_def->constant_initializers could have duplicates so make sure we only add once
    const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
    if (!function_body_graph.GetInitializedTensor(constant_initializer, subgraph_initializer)) {
      function_body_graph.AddInitializedTensor(*initializer);
    }
  }

  //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
  auto status = function_body_graph.Resolve();
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
}

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           const ONNX_NAMESPACE::FunctionProto& onnx_func,
                           const logging::Logger& logger)
    : body_(onnx_func.name(), false, onnxruntime::ModelMetaData(),
            graph.ModelPath().ToPathString(), IOnnxRuntimeOpSchemaRegistryList(),
            onnx_func.opset_import_size() != 0 ? GetFunctionOpsetImports(onnx_func, graph.DomainToVersionMap()) : graph.DomainToVersionMap(),
            {}, logger) {
}

FunctionImpl::~FunctionImpl() = default;

const onnxruntime::Graph& FunctionImpl::Body() const {
  return body_.MainGraph();
}

onnxruntime::Graph& FunctionImpl::MutableBody() {
  return body_.MainGraph();
}

}  // namespace onnxruntime
