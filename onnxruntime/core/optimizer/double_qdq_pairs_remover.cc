// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/double_qdq_pairs_remover.h"
#include <cassert>
#include <string>

#include "core/common/span_utils.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

/// <summary>
/// Returns the zero-point type from the given QuantizeLinear node.
/// </summary>
/// <param name="graph">Graph</param>
/// <param name="q_node">QuantizeLinear node</param>
/// <param name="zp_data_type">Output parameter to store the zero-point data type</param>
/// <returns>True if successfully extracted the zero-point data type</returns>
static bool GetQNodeZeroPointType(const Graph& graph, const Node& q_node,
                                  /*out*/ ONNX_NAMESPACE::TensorProto_DataType& zp_data_type) {
  assert(q_node.OpType() == "QuantizeLinear");
  const auto input_defs = q_node.InputDefs();

  if (QDQ::InputIndex::ZERO_POINT_ID >= input_defs.size() || !input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Exists()) {
    // If a zero_point input is absent, get the type from the "output_dtype" attribute or default to uint8.
    // The "output_dtype" attribute was added in ONNX opset 21.
    const auto* attr = graph_utils::GetNodeAttribute(q_node, "output_dtype");
    zp_data_type = attr != nullptr ? static_cast<ONNX_NAMESPACE::TensorProto_DataType>(attr->i())
                                   : ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    return true;
  }

  const auto* zp_proto = graph.GetConstantInitializer(input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name(), true);
  if (zp_proto == nullptr) {
    return false;
  }

  zp_data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(zp_proto->data_type());
  return true;
}

// Applies a new zero point or scale as the input for a Q/DQ node.
template <typename T>
static void ApplyNewInputValue(Graph& graph, Node& node, QDQ::InputIndex index, T value) {
  const auto* input_tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[index]->Name());
  Initializer input_init{graph, *input_tensor, graph.ModelPath()};
  ONNX_NAMESPACE::TensorProto new_input_tensor;
  input_init.data<T>()[0] = value;
  input_init.ToProto(new_input_tensor);
  auto new_name = graph.GenerateNodeArgName("DoubleQDQRemoved_" + node.InputDefs()[index]->Name());
  new_input_tensor.set_name(new_name);
  new_input_tensor.add_dims(1);
  NodeArg& new_input = graph_utils::AddInitializer(graph, new_input_tensor);
  graph_utils::ReplaceNodeInput(node, index, new_input);
}

// Returns a new zero point and scale value for the given Q/DQ nodes.
template <typename T>
static bool FindNewZeroPointAndScale(const Graph& graph, const Node& node1, const Node& node2,
                                     float& new_scale, T& new_zero_point, bool& skip_reset) {
  // scale & zero point share same initializer, no need to reset the value
  const std::string& node1_scale_name = node1.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
  const std::string& node2_scale_name = node2.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
  const std::string& node1_zp_name = node1.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
  const std::string& node2_zp_name = node2.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
  skip_reset = false;
  if (node1_scale_name == node2_scale_name && node1_zp_name == node2_zp_name) {
    skip_reset = true;
    return true;
  }
  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* node1_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node1_scale_name);
  const ONNX_NAMESPACE::TensorProto* node2_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node2_scale_name);
  const ONNX_NAMESPACE::TensorProto* node1_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node1_zp_name);
  const ONNX_NAMESPACE::TensorProto* node2_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node2_zp_name);
  Initializer zero_point_init_1{graph, *node1_zp_tensor_proto, graph.ModelPath()};
  Initializer zero_point_init_2{graph, *node2_zp_tensor_proto, graph.ModelPath()};
  Initializer scale_init_1{graph, *node1_scale_tensor_proto, graph.ModelPath()};
  Initializer scale_init_2{graph, *node2_scale_tensor_proto, graph.ModelPath()};
  if (zero_point_init_1.data_type() != zero_point_init_2.data_type() ||
      scale_init_1.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      scale_init_2.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }

  T zero_point_1 = zero_point_init_1.data<T>()[0];
  T zero_point_2 = zero_point_init_2.data<T>()[0];
  const float scale_1 = scale_init_1.data<float>()[0];
  const float scale_2 = scale_init_2.data<float>()[0];
  // No need to rest the value if values are equal
  if (zero_point_1 == zero_point_2 && abs(scale_1 - scale_2) < 1E-20) {
    skip_reset = true;
    return true;
  }
  T q_min = std::numeric_limits<T>::min();
  T q_max = std::numeric_limits<T>::max();

  float real_min1 = gsl::narrow_cast<float>(q_min - zero_point_1) * scale_1;
  float real_max1 = gsl::narrow_cast<float>(q_max - zero_point_1) * scale_1;
  float real_min2 = gsl::narrow_cast<float>(q_min - zero_point_2) * scale_2;
  float real_max2 = gsl::narrow_cast<float>(q_max - zero_point_2) * scale_2;

  const float real_min = std::max(real_min1, real_min2);
  const float real_max = std::min(real_max1, real_max2);

  new_scale = (real_max - real_min) / gsl::narrow_cast<float>(q_max - q_min);
  new_zero_point = gsl::narrow_cast<T>(std::round(gsl::narrow_cast<float>(q_min) - real_min / new_scale));
  return true;
}

// Recomputes the zero point and scale of the outer Q/DQ nodes (i.e., Q1 and DQ2(s)). This is necessary because
// the original two QDQ pairs may have different zero-points and scales. Ex: Q1 -> DQ1 -> Q2 -> DQ2*, where
// the first pair has (zp1, scale1) and the second pair has (zp2, scale2).
// After removing the middle two nodes, the zero point and scale of the final (outer) ops must be recomputed
// for correctness.
template <typename ZeroPointType>
static bool RecomputeOuterQDQZeroPointAndScale(Graph& graph, Node& q1, const Node& dq1, const Node& q2,
                                               gsl::span<gsl::not_null<Node*>> dq2s) {
  if (dq2s.empty()) {
    return false;
  }

  bool no_change_needed = false;
  float new_scale = 0.0f;
  ZeroPointType new_zero_point = 0;
  if (!FindNewZeroPointAndScale(graph, dq1, q2, new_scale, new_zero_point, no_change_needed)) {
    return false;
  }
  if (no_change_needed) {
    return true;
  }

  ApplyNewInputValue(graph, q1, QDQ::InputIndex::SCALE_ID, new_scale);
  ApplyNewInputValue(graph, q1, QDQ::InputIndex::ZERO_POINT_ID, new_zero_point);

  for (gsl::not_null<Node*> dq2 : dq2s) {
    ApplyNewInputValue(graph, *dq2, QDQ::InputIndex::SCALE_ID, new_scale);
    ApplyNewInputValue(graph, *dq2, QDQ::InputIndex::ZERO_POINT_ID, new_zero_point);
  }

  return true;
}

/// <summary>
/// Tries to reduce a double QDQ sequence (Q1 -> DQ1 -> Q2 -> DQ2*) beginning with the provided Q1 node index.
/// The scale/zero-point values of the outer Q1 and DQ2* nodes may need to be recomputed.
/// Supports multiple identical DQ2 nodes.
/// </summary>
/// <param name="graph">Graph to modify</param>
/// <param name="q1_index">Index of potential Q1 node</param>
/// <returns>True if the double QDQ sequence was reduced</returns>
static bool TryReduceDoubleQDQSequence(Graph& graph, NodeIndex q1_index) {
  const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  // Ensure that q1 is a Q operator, has only one output, and is not a graph output
  Node* q1 = graph.GetNode(q1_index);
  if (q1 == nullptr ||
      q1->OpType() != "QuantizeLinear" ||
      q1->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*q1)) {
    return false;
  }

  // Ensure that dq1 is a DQ operator, has one parent and one child, and is not a graph output
  NodeIndex dq1_index = q1->OutputEdgesBegin()->GetNode().Index();
  const Node* dq1 = graph.GetNode(dq1_index);
  if (dq1 == nullptr ||
      dq1->OpType() != "DequantizeLinear" ||
      dq1->GetInputEdgesCount() != 1 ||
      dq1->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*dq1)) {
    return false;
  }

  // The Q1 and DQ1 nodes must have equal zero-point and scale values (scalar/constant).
  if (!QDQ::IsQDQPairSupported(graph, *q1, *dq1, get_constant_initializer, graph.ModelPath())) {
    return false;
  }

  auto q1_quant_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (!GetQNodeZeroPointType(graph, *q1, q1_quant_type)) {
    return false;
  }

  // Ensure that q2 is a Q operator, its output is not a graph output, and that its zero-point quantization type
  // is equal to q1's.
  NodeIndex q2_index = dq1->OutputEdgesBegin()->GetNode().Index();
  const Node* q2 = graph.GetNode(q2_index);
  auto q2_quant_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;

  if (q2 == nullptr ||
      q2->OpType() != "QuantizeLinear" ||
      graph.NodeProducesGraphOutput(*q2) ||
      !GetQNodeZeroPointType(graph, *q2, q2_quant_type) ||
      q1_quant_type != q2_quant_type) {
    return false;
  }

  // All of q2's children should be DQ nodes with zero-point and scale values equal to those of q2.
  InlinedVector<gsl::not_null<Node*>> dq2_nodes;
  dq2_nodes.reserve(q2->GetOutputEdgesCount());

  for (auto it = q2->OutputEdgesBegin(); it != q2->OutputEdgesEnd(); it++) {
    NodeIndex dq2_index = it->GetNode().Index();
    Node* dq2 = graph.GetNode(dq2_index);

    if (dq2 == nullptr || dq2->OpType() != "DequantizeLinear") {
      // Child is not a DQ op.
      return false;
    }

    // The Q2 and DQ2 nodes must have equal zero-point and scale values (scalar/constant).
    if (!QDQ::IsQDQPairSupported(graph, *q2, *dq2, get_constant_initializer, graph.ModelPath())) {
      return false;
    }

    dq2_nodes.push_back(dq2);
  }

  bool can_recompute = false;
  if (q1_quant_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    can_recompute = RecomputeOuterQDQZeroPointAndScale<uint8_t>(graph, *q1, *dq1, *q2, dq2_nodes);
  } else if (q1_quant_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    can_recompute = RecomputeOuterQDQZeroPointAndScale<int8_t>(graph, *q1, *dq1, *q2, dq2_nodes);
  } else if (q1_quant_type == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    can_recompute = RecomputeOuterQDQZeroPointAndScale<uint16_t>(graph, *q1, *dq1, *q2, dq2_nodes);
  } else if (q1_quant_type == ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    can_recompute = RecomputeOuterQDQZeroPointAndScale<int16_t>(graph, *q1, *dq1, *q2, dq2_nodes);
  }

  if (!can_recompute) {
    return false;
  }

  graph.RemoveEdge(q1_index, dq1_index, 0, 0);  // Disconnect Q1 -> DQ1
  graph.RemoveEdge(dq1_index, q2_index, 0, 0);  // Disconnect DQ1 -> Q2

  // Disconnect Q2 --> DQ2(s)
  // Connect Q1 -> DQ2(s)
  for (gsl::not_null<Node*> dq2 : dq2_nodes) {
    graph.RemoveEdge(q2_index, dq2->Index(), 0, 0);
    graph.AddEdge(q1_index, dq2->Index(), 0, 0);
  }

  graph.RemoveNode(q2_index);
  graph.RemoveNode(dq1_index);

  return true;
}

Status DoubleQDQPairsRemover::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  const GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex node_index : node_topology_list) {
    if (TryReduceDoubleQDQSequence(graph, node_index)) {
      modified = true;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
