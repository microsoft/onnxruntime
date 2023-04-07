// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/qdq_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

namespace {

int ReplaceOrCreateZeroPointInitializer(Graph& graph, Node& quantize_node) {
  // Create a new zero point initializer which has float type.
  // Replace the input of the quantize node with this new zero point initializer
  // or add this initializer as an input to the quantize node if the input did
  // not previously exist.

  const auto quant_node_input_defs = quantize_node.MutableInputDefs();
  int zero_point_type = ONNX_NAMESPACE::TensorProto_DataType_INT8;
  ONNX_NAMESPACE::TensorProto zero_point_tensor_float;
  if (quant_node_input_defs.size() >= 3) {
    // The quantize node has the zero point input
    auto zero_point_tensor_int = graph.GetInitializer(quant_node_input_defs[2]->Name(), true);
    ORT_ENFORCE(zero_point_tensor_int != nullptr, "Expected: zero point initializer with name ",
                quant_node_input_defs[2]->Name(), " to be present in the graph. Actual: not found.");
    zero_point_type = zero_point_tensor_int->data_type();
    zero_point_tensor_float.set_name(graph.GenerateNodeArgName(zero_point_tensor_int->name()));
    zero_point_tensor_float.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (const auto val : zero_point_tensor_int->int32_data()) {
      zero_point_tensor_float.add_float_data(static_cast<float>(val));
    }
    for (const auto& dim : zero_point_tensor_int->dims()) {
      zero_point_tensor_float.add_dims(dim);
    }
    graph.RemoveInitializedTensor(zero_point_tensor_int->name());

    // Since the quantize node has the zero point initializer input, replace it
    graph_utils::ReplaceNodeInput(quantize_node, 2, graph_utils::AddInitializer(graph, zero_point_tensor_float));
  } else {
    // The quantize node does not have the zero point optional input.
    // Create the zero point initializer to be 0.
    zero_point_tensor_float.set_name(graph.GenerateNodeArgName("zero_point"));
    zero_point_tensor_float.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    zero_point_tensor_float.add_float_data(0.0f);

    // Since the input did not exist, add the newly created initializer as an input
    graph_utils::AddNodeInput(quantize_node, 2, graph_utils::AddInitializer(graph, zero_point_tensor_float));
  }

  return zero_point_type;
}

void FuseQDQNodes(Graph& graph, Node& quantize_node, Node& dequantize_node, int quant_type) {
  // Prepare the FakeQuant node that will be the replacement node during fusion.
  Node& fake_quant_node = graph.AddNode(graph.GenerateNodeName("FakeQuant"), "FakeQuant",
                                        "FakeQuant after fusion of " + quantize_node.Name() + " and " +
                                            dequantize_node.Name(),
                                        quantize_node.MutableInputDefs(), {}, {}, kMSDomain);

  // The quant_min and quant_max attribute values are set based on the elem_type of the zero point tensor.
  // If the type is INT8, the quant_min and quant_max are set to -128 and 127 respectively. Else UINT8 is
  // assumed, and quant_min and quant_max are set to 0 and 255 respectively.
  fake_quant_node.AddAttribute(
      "quant_min", static_cast<int64_t>(quant_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 ? -128 : 0));
  fake_quant_node.AddAttribute(
      "quant_max", static_cast<int64_t>(quant_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 ? 127 : 255));

  fake_quant_node.SetExecutionProviderType(quantize_node.GetExecutionProviderType());

  // Fuse the Q->DQ into FakeQuant
  graph_utils::FinalizeNodeFusion(graph, {quantize_node, dequantize_node}, fake_quant_node);

  // Since FakeQuant needs an extra output, add it now since fusion is complete.
  ONNX_NAMESPACE::TypeProto bool_mask_tensor;
  bool_mask_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  NodeArg& fake_quant_gradient_mask =
      graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("fake_quant_gradient_mask"), &bool_mask_tensor);
  fake_quant_node.MutableOutputDefs().emplace_back(&fake_quant_gradient_mask);
}

std::pair<bool, Node*> CheckForQDQPatternMatch(Graph& graph, Node& quantize_node,
                                               const InlinedHashSet<std::string_view>& compatible_execution_providers) {
  // Try to match the current node with QuantizeLinear in the effort of searching for the pattern
  // QuantizeLinear -> DequantizeLinear.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(quantize_node, "QuantizeLinear", {10, 13}) ||
      !graph_utils::IsSupportedProvider(quantize_node, compatible_execution_providers)) {
    return {false, nullptr};
  }

  // QuantizeLinear node found. Assert that there is only one node consuming the output of the QuantizeLinear node.
  ORT_ENFORCE(quantize_node.GetOutputEdgesCount() == 1, "There must only be a single node consuming the output of ",
              "QuantizeLinear node. Actual: The output of QuantizeLinear (", quantize_node.Name(),
              ") is being consumed by ", quantize_node.GetOutputEdgesCount(), " nodes.");

  // Now try to match the succeeding node with DequantizeLinear to ascertain QuantizeLinear -> DequantizeLinear
  // pattern.
  Node* dequantize_node_ptr = graph.GetNode(quantize_node.OutputNodesBegin()->Index());

  // Every Q must be followed by a single and unique DQ and every DQ must be preceded by a single and unique Q.
  // And hence every Q->DQ pair can be fused into a single and unique FakeQuant.
  // So, we error out and let the user know that the graph is not what is expected if DequantizeLinear is not found.
  ORT_ENFORCE(dequantize_node_ptr, "Expected: A DequantizeLinear node, Actual: nullptr.");
  ORT_ENFORCE(graph_utils::IsSupportedOptypeVersionAndDomain(*dequantize_node_ptr, "DequantizeLinear", {10, 13}) &&
                  graph_utils::IsSupportedProvider(*dequantize_node_ptr, compatible_execution_providers),
              "Expected that every QuantizeLinear node be followed by a unique DequantizeLinear node. ",
              "Actual: QuantizeLinear (", quantize_node.Name(), ") is followed by ", dequantize_node_ptr->OpType(), "(",
              dequantize_node_ptr->Name(), ").");

  return {true, dequantize_node_ptr};
}

}  // namespace

Status QDQFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& nodes_in_topological_order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : nodes_in_topological_order) {
    auto* quantize_node_ptr = graph.GetNode(node_index);
    if (!quantize_node_ptr) continue;  // Node was removed.

    auto& quantize_node = *quantize_node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(quantize_node, modified, graph_level, logger));

    // Check for QDQ pattern
    auto [qdq_pattern_found, dequantize_node_ptr] =
        CheckForQDQPatternMatch(graph, quantize_node, GetCompatibleExecutionProviders());
    if (!qdq_pattern_found) {
      continue;
    }

    // QuantizeLinear zero_point is INT8 or UINT8. FakeQuant uses quant_zero_point as FLOAT.
    // So, remove the old initializers and update the zero point to be of FLOAT type if it exists.
    // If the initializer does not exist, create a new zero point initializer with the correct type.
    const auto quant_type = ReplaceOrCreateZeroPointInitializer(graph, quantize_node);

    // Fuse the QDQ pattern into FakeQuant and move the inputs and outputs.
    FuseQDQNodes(graph, quantize_node, *dequantize_node_ptr, quant_type);

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
