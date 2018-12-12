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
  // This will most likely be done with a Transformer rather than a RewriteRule, so it needs
  // mode thought.
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
  // Disable default graph transformers for the constant node.
  so.enable_default_transformers = false;
  InferenceSession session_object{so};
  // TODO Can we directly pass the model to the session object instead of having to dump it to a stream?
  std::stringstream model_istream;
  p_model->ToProto().SerializeToOstream(&model_istream);
  ONNXRUNTIME_RETURN_IF_ERROR(session_object.Load(model_istream));
  ONNXRUNTIME_RETURN_IF_ERROR(session_object.Initialize());

  // Prepare outputs.
  std::vector<std::string> output_names;
  for (auto& output : subgraph.GetOutputs()) {
    output_names.push_back(output->Name());
  }
  // No inputs needed as they are all initializers.
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<MLValue> fetches;

  // Execute the subgraph.
  Status st = session_object.Run(feeds, output_names, &fetches);

  // Go over all output node args and substitute them with the computed initializers.
  ONNXRUNTIME_ENFORCE(fetches.size() == node.OutputDefs().size());
  for (int fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
    MLValue& mlvalue = fetches[fetch_idx];
    if (mlvalue.Fence()) {  // TODO Is this needed?
      mlvalue.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
    }
    // TODO Can it be anything else than a tensor here?
    ONNXRUNTIME_ENFORCE(mlvalue.IsTensor());
    const Tensor& out_tensor = mlvalue.Get<Tensor>();

    // The output arg of the constant subgraph (i.e., node) that was precomputed.
    const auto* constant_arg_out = subgraph.GetOutputs()[fetch_idx];

    // Build the TensorProto that will be added as an initializer.
    ONNX_NAMESPACE::TensorProto out_tensorproto;
    out_tensorproto.set_name(constant_arg_out->Name());

    for (auto& dim : out_tensor.Shape().GetDims()) {
      out_tensorproto.add_dims(dim);
    }
    auto tensorproto_type = constant_arg_out->TypeAsProto()->tensor_type().elem_type();

    out_tensorproto.set_data_type(tensorproto_type);
    auto tensor_shape_size = out_tensor.Shape().Size();
    auto data_size = out_tensor.DataType()->Size() * tensor_shape_size;
    out_tensorproto.set_raw_data(out_tensor.DataRaw(out_tensor.DataType()), data_size);

    graph.AddInitializedTensor(out_tensorproto);

    /*ONNXRUNTIME_ENFORCE(tensorproto_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    const float* float_data = out_tensor.Data<float>();
    for (int i = 0; i < tensor_shape_size; i++) {
      float dat = float_data[i];
      out_tensorproto.add_float_data(dat);
    }*/

    // Remove the output edges of the constant node.
    std::vector<onnxruntime::NodeIndex> edge_nodes_to_remove;
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      edge_nodes_to_remove.push_back((*it).Index());
    }

    const auto* node_out_arg = node.OutputDefs()[fetch_idx];
    for (auto& edge_node_idx : edge_nodes_to_remove) {
      graph.RemoveEdge(node.Index(), edge_node_idx, *node_out_arg);
    }
  }

  // Remove the constant node.
  graph.RemoveNode(node.Index());

  // The output nodes already have the right input arg, since we used the same name in the initializer.
  // We could remove unused graph initializers here, but Graph::Resolve() will do it.

  modified = true;

  return Status::OK();
}  // namespace onnxruntime

bool ConstantFolding::SatisfyCondition(const Graph& graph, const Node& node) {
  return graph_edit_utils::IsConstantInputsNode(graph, node);
}

}  // namespace onnxruntime
