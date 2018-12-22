// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constant_folding.h"
#include "core/graph/graph_utils.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

Status ConstantFolding::Apply(Graph& graph, Node& node, bool& modified) {
  // TODO Can we reuse the model across calls of the constant folding rule?
  // TODO Instead of computing one node at a time, we can traverse the whole graph and
  // find constant parts of it. Then we can create bigger subgraphs to compute directly.
  // This will be a useful optimization if we expect big subgraphs to be constant, and
  // should be done through a Transformer, not a RewriteRule.
  auto p_model = std::make_unique<onnxruntime::Model>("ConstantFoldingModel", false, ModelMetaData(),
                                                      IOnnxRuntimeOpSchemaRegistryList({graph.GetSchemaRegistry()}),
                                                      graph.DomainToVersionMap());
  Graph& subgraph = p_model->MainGraph();

  std::vector<onnxruntime::NodeIndex> subgraph_nodes;
  subgraph_nodes.push_back(node.Index());

  // Build the subgraph.
  graph_edit_utils::BuildSubgraph(graph, subgraph_nodes, subgraph);

  SessionOptions so;
  so.session_logid = "ConstantFoldingSession";
  // Disable default graph transformers for the constant node (can lead to infinite recursion otherwise).
  so.enable_default_transformers = false;
  InferenceSession session_object{so};
  // TODO Make it possible to pass the model directly to the session instead of having to dump it to a stream.
  std::stringstream model_istream;
  p_model->ToProto().SerializeToOstream(&model_istream);
  ORT_RETURN_IF_ERROR(session_object.Load(model_istream));
  ORT_RETURN_IF_ERROR(session_object.Initialize());

  // Execute the subgraph. No inputs are needed as they are all initializers.
  std::vector<std::string> output_names;
  for (auto& output : subgraph.GetOutputs()) {
    output_names.push_back(output->Name());
  }
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<MLValue> fetches;

  Status st = session_object.Run(feeds, output_names, &fetches);

  // Go over all output node args and substitute them with the newly computed tensors, which will be
  // added to the graph as initializers.
  ORT_ENFORCE(fetches.size() == node.OutputDefs().size());
  for (int fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
    MLValue& mlvalue = fetches[fetch_idx];
    if (mlvalue.Fence()) {
      mlvalue.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
    }

    // Build the TensorProto that corresponds to the computed MLValue and add it as initializer to the graph.
    ONNX_NAMESPACE::TensorProto out_tensorproto;
    const auto* constant_arg_out = subgraph.GetOutputs()[fetch_idx];
    BuildTensorProtoForInitializer(mlvalue, *constant_arg_out, out_tensorproto);

    graph.AddInitializedTensor(out_tensorproto);
  }

  // Remove the output edges of the constant node and then remove the node itself.
  graph_edit_utils::RemoveNodeOutputEdges(graph, node);
  graph.RemoveNode(node.Index());

  // The output nodes already have the right input arg, since we used the same name in the initializer.
  // We could remove unused graph initializers here, but Graph::Resolve() will take care of it.

  modified = true;

  return Status::OK();
}  // namespace onnxruntime

bool ConstantFolding::SatisfyCondition(const Graph& graph, const Node& node) {
  return graph_edit_utils::IsConstantInputsNode(graph, node);
}

void ConstantFolding::BuildTensorProtoForInitializer(const MLValue& mlvalue,
                                                     const NodeArg& constant_node_arg,
                                                     ONNX_NAMESPACE::TensorProto& tensorproto) {
  ORT_ENFORCE(mlvalue.IsTensor());
  const Tensor& out_tensor = mlvalue.Get<Tensor>();

  // Set name, dimensions, type, and data of the TensorProto.
  tensorproto.set_name(constant_node_arg.Name());

  for (auto& dim : out_tensor.Shape().GetDims()) {
    tensorproto.add_dims(dim);
  }
  auto tensorproto_type = constant_node_arg.TypeAsProto()->tensor_type().elem_type();

  tensorproto.set_data_type(tensorproto_type);
  auto tensor_shape_size = out_tensor.Shape().Size();
  auto data_size = out_tensor.DataType()->Size() * tensor_shape_size;
  tensorproto.set_raw_data(out_tensor.DataRaw(out_tensor.DataType()), data_size);
}

}  // namespace onnxruntime
