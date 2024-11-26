// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/bias_quantization.h"

#include "core/common/common.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

Status BiasQuantization::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    const auto& input_defs = node.InputDefs();

    // It's Conv/Gemm node with an initializer bias.
    if ((node.OpType() != "Conv" && node.OpType() != "Gemm") || input_defs.size() < 3 || !input_defs[2]->Exists() ||
        !graph_utils::IsInitializer(graph, input_defs[2]->Name(), true)) {
      continue;
    }

    auto bias_shape = input_defs[2]->Shape();
    if (!bias_shape || bias_shape->dim_size() != 1) {
      continue;
    }
    int64_t bias_size = bias_shape->dim(0).dim_value();

    // input_0 and input_1 are outputs of DequantizeLinear nodes.
    const Node* parent_node_0 = graph.GetProducerNode(input_defs[0]->Name());
    const Node* parent_node_1 = graph.GetProducerNode(input_defs[1]->Name());
    if (!parent_node_0 || !parent_node_1 || parent_node_0->OpType() != QDQ::DQOpName ||
        parent_node_1->OpType() != QDQ::DQOpName) {
      continue;
    }

    Node& dq_0 = *graph.GetNode(parent_node_0->Index());
    Node& dq_1 = *graph.GetNode(parent_node_1->Index());

    // Currently we require input_0 is per-tensor scale.
    if (!optimizer_utils::IsScalar(*dq_0.InputDefs()[1])) {
      continue;
    }

    // For input_1, it's either per-tensor scale or per-channel scale on specific axis (0 for Conv and 1 for Gemm).
    bool is_per_tensor_scale = true;
    if (!optimizer_utils::IsScalar(*dq_1.InputDefs()[1])) {
      is_per_tensor_scale = false;
      auto weight_scale_shape = dq_1.InputDefs()[1]->Shape();
      if (!weight_scale_shape || weight_scale_shape->dim_size() != 1 || !weight_scale_shape->dim(0).has_dim_value() ||
          weight_scale_shape->dim(0).dim_value() != bias_size) {
        continue;
      }

      const auto& dq_attrs = dq_1.GetAttributes();
      if (dq_attrs.find("block_size") != dq_attrs.end()) {
        continue;
      }

      int64_t axis = 1;
      if (dq_attrs.find("axis") != dq_attrs.end()) {
        axis = dq_attrs.at("axis").i();
      }

      int64_t expected_axis = 0;
      if (node.OpType() == "Gemm") {
        int64_t transB = 0;
        if (const auto& attr = node.GetAttributes().find("transB"); attr != node.GetAttributes().end()) {
          transB = attr->second.i();
        }
        expected_axis = transB == 0 ? 1 : 0;
      }

      if (axis != expected_axis) {
        continue;
      }
    }

    // Bias is quantized to int32.
    ONNX_NAMESPACE::TypeProto int32_type_proto;
    int32_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto scale_type = dq_1.InputDefs()[1]->TypeAsProto();  // Maybe per-tensor (scalar) or per-channel (1D) scale.
    ONNX_NAMESPACE::TypeProto bias_dq_type;
    bias_dq_type.mutable_tensor_type()->set_elem_type(scale_type->tensor_type().elem_type());
    bias_dq_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(bias_size);

    // scale = input_scale_0 * input_scale_1.
    NodeArg& scale_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_scale"), scale_type);
    Node& mul_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_scale"), "Mul", "Scale node",
                                   {dq_0.MutableInputDefs()[1], dq_1.MutableInputDefs()[1]}, {&scale_node_arg}, nullptr,
                                   node.Domain());

    // fp_bias / scale.
    NodeArg& bias_div_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div"), &bias_dq_type);
    Node& div_node =
        graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div"), "Div", "Bias div node",
                      {node.MutableInputDefs()[2], &scale_node_arg}, {&bias_div_node_arg}, nullptr, node.Domain());
    graph.AddEdge(mul_node.Index(), div_node.Index(), 0, 1);

    // Round(fp_bias / scale).
    NodeArg& bias_div_round_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div_round"), &bias_dq_type);
    Node& round_node =
        graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div_round"), "Round", "Bias div round node",
                      {&bias_div_node_arg}, {&bias_div_round_node_arg}, nullptr, node.Domain());
    graph.AddEdge(div_node.Index(), round_node.Index(), 0, 0);

    // Cast(round(fp_bias / scale)) to int32.
    NodeArg& bias_int32_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_int32"), &int32_type_proto);
    Node& cast_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_int32"), "Cast", "Bias int32 node",
                                    {&bias_div_round_node_arg}, {&bias_int32_node_arg}, nullptr, node.Domain());
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
    graph.AddEdge(round_node.Index(), cast_node.Index(), 0, 0);

    // Bias DQ node produces output to Conv/Gemm node's input_2, with scale = input_scale_0 * input_scale_1, zp = 0.
    NodeArg& bias_dq_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_dq"), &bias_dq_type);
    Node& dq_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_dq"), QDQ::DQOpName, "Bias DQ node",
                                  {&bias_int32_node_arg, &scale_node_arg}, {&bias_dq_node_arg}, nullptr, node.Domain());
    if (!is_per_tensor_scale) {
      dq_node.AddAttribute("axis", static_cast<int64_t>(0));
    }

    graph.AddEdge(cast_node.Index(), dq_node.Index(), 0, 0);
    graph.AddEdge(mul_node.Index(), dq_node.Index(), 0, 1);
    node.MutableInputDefs()[2] = &bias_dq_node_arg;
    graph.AddEdge(dq_node.Index(), node.Index(), 0, 2);

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
