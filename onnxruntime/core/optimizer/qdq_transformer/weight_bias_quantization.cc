// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/weight_bias_quantization.h"

#include "core/common/common.h"
#include "core/util/qmath.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

Status WeightBiasQuantization::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.OpType() != "Conv" && node.OpType() != "ConvTranspose" && node.OpType() != "Gemm") {
      continue;
    }

    const auto& input_defs = node.InputDefs();
    const NodeArg* input_arg = input_defs[0];
    const NodeArg* weight_arg = input_defs[1];
    const NodeArg* bias_arg = input_defs.size() >= 3 && input_defs[2]->Exists() ? input_defs[2] : nullptr;
    const Node* parent_node_0 = graph.GetProducerNode(input_arg->Name());
    const Node* parent_node_1 = graph.GetProducerNode(weight_arg->Name());

    // Currently we require input is Dequantized with per-tensor scale.
    if (!parent_node_0 || parent_node_0->OpType() != QDQ::DQOpName ||
        !optimizer_utils::IsScalar(*parent_node_0->InputDefs()[1])) {
      continue;
    }

    Node& dq_0 = *graph.GetNode(parent_node_0->Index());
    Node* dq_1 = nullptr;
    const ONNX_NAMESPACE::TensorProto* weight_proto = nullptr;
    if (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName) {
      dq_1 = graph.GetNode(parent_node_1->Index());
    } else if (!graph_utils::IsInitializer(graph, weight_arg->Name(), true) ||
               !graph.GetInitializedTensor(weight_arg->Name(), weight_proto) ||
               weight_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      // Support float32 weight initializer only for now.
      continue;
    }

    int64_t bias_size = -1;
    if (bias_arg) {
      auto bias_shape = bias_arg->Shape();
      if (!graph_utils::IsInitializer(graph, bias_arg->Name(), true) || !bias_shape || bias_shape->dim_size() != 1) {
        continue;
      }

      bias_size = bias_shape->dim(0).dim_value();
    }

    // Nothing to do if neither weight nor bias is initializer.
    if (dq_1 && bias_size == -1) {
      continue;
    }

    bool is_per_tensor_scale = true;
    // If weight is quantized, it's either per-tensor or per-channel on specific axis (0 for Conv and 1 for Gemm).
    if (dq_1 && !optimizer_utils::IsScalar(*dq_1->InputDefs()[1])) {
      is_per_tensor_scale = false;
      auto weight_scale_shape = dq_1->InputDefs()[1]->Shape();
      if (!weight_scale_shape || weight_scale_shape->dim_size() != 1 || !weight_scale_shape->dim(0).has_dim_value() ||
          weight_scale_shape->dim(0).dim_value() != bias_size) {
        continue;
      }

      const auto& dq_attrs = dq_1->GetAttributes();
      if (dq_attrs.find("block_size") != dq_attrs.end()) {
        continue;
      }

      int64_t axis = 1;
      if (auto axis_iter = dq_attrs.find("axis"); axis_iter != dq_attrs.end()) {
        axis = axis_iter->second.i();
      }

      int64_t expected_axis = 0;
      if (node.OpType() == "Gemm") {
        int64_t transB = 0;
        const auto& gemm_attrs = node.GetAttributes();
        if (auto trans_b_iter = gemm_attrs.find("transB"); trans_b_iter != gemm_attrs.end()) {
          transB = trans_b_iter->second.i();
        }
        expected_axis = transB == 0 ? 1 : 0;
      }

      if (axis != expected_axis) {
        continue;
      }
    }

    NodeArg* weight_scale_arg = nullptr;
    if (!dq_1) {
      auto initializer = std::make_unique<Initializer>(*weight_proto, graph.ModelPath());
      const float* weight_data = initializer->data<float>();

      // Quantize float32 weight to int8_t (per-tensor, symmetric).
      // int8_t quantization of input[1] works with input[0] of all types.
      float scale;
      int8_t zp;
      GetQuantizationParameter(weight_data, static_cast<int64_t>(initializer->size()), scale, zp, nullptr);

      // Weight scale initializer.
      ONNX_NAMESPACE::TensorProto weight_scale_proto;
      weight_scale_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_weight_scale"));
      weight_scale_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      weight_scale_proto.mutable_float_data()->Add(scale);
      weight_scale_arg = &graph_utils::AddInitializer(graph, weight_scale_proto);

      // Weight zero point initializer.
      ONNX_NAMESPACE::TensorProto weight_zp_proto;
      weight_zp_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_weight_zp"));
      weight_zp_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
      weight_zp_proto.mutable_int32_data()->Add(static_cast<int32_t>(zp));
      NodeArg& weight_zp_arg = graph_utils::AddInitializer(graph, weight_zp_proto);

      // Q from float32 to int8.
      ONNX_NAMESPACE::TypeProto weight_q_type_proto;
      weight_q_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
      *weight_q_type_proto.mutable_tensor_type()->mutable_shape() = *weight_arg->Shape();
      NodeArg& weight_q_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_weight_q"), &weight_q_type_proto);
      Node& weight_q_node = graph.AddNode(
          graph.GenerateNodeArgName(node.Name() + "_weight_q"), QDQ::QOpName, "Weight Q node",
          {node.MutableInputDefs()[1], weight_scale_arg, &weight_zp_arg}, {&weight_q_arg}, nullptr, node.Domain());

      // DQ from int8 to float32.
      NodeArg& weight_dq_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_weight_dq"), weight_arg->TypeAsProto());
      Node& weight_dq_node =
          graph.AddNode(graph.GenerateNodeArgName(node.Name() + "_weight_dq"), QDQ::DQOpName, "Weight DQ node",
                        {&weight_q_arg, weight_scale_arg, &weight_zp_arg}, {&weight_dq_arg}, nullptr, node.Domain());
      graph.AddEdge(weight_q_node.Index(), weight_dq_node.Index(), 0, 0);
      node.MutableInputDefs()[1] = &weight_dq_arg;
      graph.AddEdge(weight_dq_node.Index(), node.Index(), 0, 1);
    } else {
      weight_scale_arg = dq_1->MutableInputDefs()[1];
    }

    if (bias_size != -1) {
      // Bias is quantized to int32. Q cannot support int32 as target type, need to compose the whole computation.
      // bias_scale = input_scale * weight_scale.
      NodeArg& bias_scale_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_scale"),
                                                         weight_scale_arg->TypeAsProto());
      Node& mul_node =
          graph.AddNode(graph.GenerateNodeName(node.Name() + "_scale"), "Mul", "Bias scale node",
                        {dq_0.MutableInputDefs()[1], weight_scale_arg}, {&bias_scale_arg}, nullptr, node.Domain());

      // fp_bias / scale.
      NodeArg& bias_div_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div"), bias_arg->TypeAsProto());
      Node& div_node =
          graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div"), "Div", "Bias div node",
                        {node.MutableInputDefs()[2], &bias_scale_arg}, {&bias_div_arg}, nullptr, node.Domain());
      graph.AddEdge(mul_node.Index(), div_node.Index(), 0, 1);

      // Round(fp_bias / scale).
      NodeArg& bias_div_round_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div_round"), bias_arg->TypeAsProto());
      Node& round_node =
          graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div_round"), "Round", "Bias div round node",
                        {&bias_div_arg}, {&bias_div_round_arg}, nullptr, node.Domain());
      graph.AddEdge(div_node.Index(), round_node.Index(), 0, 0);

      // Cast(Round(fp_bias / scale)) to int32.
      ONNX_NAMESPACE::TypeProto bias_int32_type_proto;
      bias_int32_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
      *bias_int32_type_proto.mutable_tensor_type()->mutable_shape() = *bias_arg->Shape();
      NodeArg& bias_int32_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_int32"), &bias_int32_type_proto);
      Node& cast_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_int32"), "Cast", "Bias INT32 node",
                                      {&bias_div_round_arg}, {&bias_int32_arg}, nullptr, node.Domain());
      cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
      graph.AddEdge(round_node.Index(), cast_node.Index(), 0, 0);

      // Bias DQ node produces output to Conv/Gemm node's input_2, with scale = input_scale_0 * input_scale_1, zp = 0.
      NodeArg& bias_dq_arg =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_dq"), bias_arg->TypeAsProto());
      Node& bias_dq_node =
          graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_dq"), QDQ::DQOpName, "Bias DQ node",
                        {&bias_int32_arg, &bias_scale_arg}, {&bias_dq_arg}, nullptr, node.Domain());
      if (!is_per_tensor_scale) {
        bias_dq_node.AddAttribute("axis", static_cast<int64_t>(0));
      }

      graph.AddEdge(cast_node.Index(), bias_dq_node.Index(), 0, 0);
      graph.AddEdge(mul_node.Index(), bias_dq_node.Index(), 0, 1);
      node.MutableInputDefs()[2] = &bias_dq_arg;
      graph.AddEdge(bias_dq_node.Index(), node.Index(), 0, 2);
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
