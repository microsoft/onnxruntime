// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/constant_folding.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

Status ConstantFolding::Apply(Graph& graph, Node& node, bool& modified, bool& deleted) {
  // Create execution frame for executing constant nodes.
  OptimizerExecutionFrame::Info info({&node}, graph.GetAllInitializedTensors());

  std::vector<int> fetch_mlvalue_idxs;
  for (const auto* node_out : node.OutputDefs()) {
    fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
  }

  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

  auto* kernel = info.GetKernel(node.Index());
  OpKernelContext op_kernel_context(&frame, kernel, ::onnxruntime::logging::LoggingManager::DefaultLogger());

  kernel->Compute(&op_kernel_context);

  std::vector<MLValue> fetches;
  frame.GetOutputs(fetches);

  // Go over all output node args and substitute them with the newly computed tensors, which will be
  // added to the graph as initializers.
  ORT_ENFORCE(fetches.size() == node.OutputDefs().size());
  for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
    MLValue& mlvalue = fetches[fetch_idx];

    // Build the TensorProto that corresponds to the computed MLValue and add it as initializer to the graph.
    ONNX_NAMESPACE::TensorProto out_tensorproto;
    const auto* constant_arg_out = node.OutputDefs()[fetch_idx];
    BuildTensorProtoForInitializer(mlvalue, *constant_arg_out, out_tensorproto);

    graph.AddInitializedTensor(out_tensorproto);
  }

  // Remove the output edges of the constant node and then remove the node itself.
  graph_utils::RemoveNodeOutputEdges(graph, node);
  graph.RemoveNode(node.Index());

  // The output nodes already have the right input arg, since we used the same name in the initializer.
  // We could remove unused graph initializers here, but Graph::Resolve() will take care of it.

  modified = deleted = true;

  return Status::OK();
}  // namespace onnxruntime

bool ConstantFolding::SatisfyCondition(const Graph& graph, const Node& node) {
  return excluded_op_types_.find(node.OpType()) == excluded_op_types_.end() &&
         // constant folding is not currently supported in subgraphs.
         node.ImplicitInputDefs().size() == 0 &&
         graph_utils::AllNodeInputsAreConstant(graph, node);
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
