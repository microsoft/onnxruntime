// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/double_qdq_pairs_remover.h"
#include <cassert>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

// Applies a new zero point or scale as the input for a Q/DQ node.
template <typename T>
static void ApplyNewInputValue(Graph& graph, Node& node, QDQ::InputIndex index, T value) {
  const auto* input_tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[index]->Name());
  Initializer input_init{*input_tensor, graph.ModelPath()};
  ONNX_NAMESPACE::TensorProto new_input_tensor(*input_tensor);
  input_init.data<T>()[0] = value;
  input_init.ToProto(new_input_tensor);
  auto new_name = graph.GenerateNodeArgName("DoubleQDQRemoved_" + node.InputDefs()[index]->Name());
  new_input_tensor.set_name(new_name);
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
  Initializer zero_point_init_1{*node1_zp_tensor_proto, graph.ModelPath()};
  Initializer zero_point_init_2{*node2_zp_tensor_proto, graph.ModelPath()};
  Initializer scale_init_1{*node1_scale_tensor_proto, graph.ModelPath()};
  Initializer scale_init_2{*node2_scale_tensor_proto, graph.ModelPath()};
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

// Recomputes the zero point and scale of the outer Q/DQ nodes (i.e., Q1 and DQ2). This is necessary because
// the original two QDQ pairs may have different zero-points and scales. Ex: Q1 -> DQ1 -> Q2 -> DQ2, where
// the first pair has (zp1, scale1) and the second pair has (zp2, scale2).
// After removing the middle two nodes, the zero point and scale of the final (outer) ops must be recomputed
// for correctness.
template <typename ZeroPointType>
static bool RecomputeOuterQDQZeroPointAndScale(Graph& graph, Node& q1, const Node& dq1, const Node& q2, Node& dq2) {
  bool skip_reset = false;
  float new_scale = 0.0f;
  ZeroPointType new_zero_point = 0;
  if (!FindNewZeroPointAndScale(graph, dq1, q2, new_scale, new_zero_point, skip_reset)) {
    return false;
  }
  if (skip_reset) {
    return true;
  }
  ApplyNewInputValue(graph, dq2, QDQ::InputIndex::SCALE_ID, new_scale);
  ApplyNewInputValue(graph, q1, QDQ::InputIndex::SCALE_ID, new_scale);
  ApplyNewInputValue(graph, dq2, QDQ::InputIndex::ZERO_POINT_ID, new_zero_point);
  ApplyNewInputValue(graph, q1, QDQ::InputIndex::ZERO_POINT_ID, new_zero_point);

  return true;
}

// Checks if the provided node index (dq1_index) is a part of a valid double QDQ pair sequence
// (i.e., Q1 -> DQ1 -> Q2 -> DQ2) that can be reduced to the outer Q/DQ nodes (i.e., Q1 -> DQ2).
// If so, the zero point and scale of the outer Q/DQ nodes are recomputed and the node indices of the other nodes
// in the sequence (i.e., Q1, Q2, and DQ2) are returned via output parameters.
static bool IsReducibleDoubleQDQSequence(Graph& graph, NodeIndex& q1_index, NodeIndex dq1_index,
                                         NodeIndex& q2_index, NodeIndex& dq2_index) {
  // Ensure that dq1 is a DQ operator, has one parent and one child, and is not a graph output
  Node* dq1 = graph.GetNode(dq1_index);
  if (dq1 == nullptr ||
      dq1->OpType() != "DequantizeLinear" ||
      dq1->GetInputEdgesCount() != 1 ||
      dq1->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*dq1)) {
    return false;
  }

  // Ensure that q2 is a Q operator, has only one child, and is not a graph output
  q2_index = dq1->OutputEdgesBegin()->GetNode().Index();
  const Node* q2 = graph.GetNode(q2_index);
  if (q2 == nullptr ||
      q2->OpType() != "QuantizeLinear" ||
      q2->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*q2)) {
    return false;
  }

  // Ensure that q1 is a Q operator, has only one output, and is not a graph output
  q1_index = dq1->InputEdgesBegin()->GetNode().Index();
  Node* q1 = graph.GetNode(q1_index);
  if (q1 == nullptr ||
      q1->GetOutputEdgesCount() != 1 ||
      q1->OpType() != "QuantizeLinear" ||
      graph.NodeProducesGraphOutput(*q1)) {
    return false;
  }

  // Ensure the dq2 is a DQ operator.
  dq2_index = q2->OutputEdgesBegin()->GetNode().Index();
  Node* dq2 = graph.GetNode(dq2_index);
  if (dq2 == nullptr ||
      dq2->OpType() != "DequantizeLinear") {
    return false;
  }

  const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  // Each QDQ pair (i.e., q1 -> dq1, q2 -> dq2) has to meet the following additional requirements:
  // - Scalar/constant zero-point and scale.
  // - The DQ and Q ops within a pair must have the same scale and zero-point.
  //   However, each pair is allowed to have different scales and zero-points.
  //
  // TODO: IsQDQPairSupported() requires an explicit zero-point input, but technically a default
  // value of 0 could be fine.
  if (!QDQ::IsQDQPairSupported(*q1, *dq1, get_constant_initializer, graph.ModelPath()) ||
      !QDQ::IsQDQPairSupported(*q2, *dq2, get_constant_initializer, graph.ModelPath())) {
    return false;
  }

  const auto& dq1_input_defs = dq1->InputDefs();
  const ONNX_NAMESPACE::TensorProto* dq1_zp_tensor_proto = graph.GetConstantInitializer(
      dq1_input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name(), true);

  assert(dq1_zp_tensor_proto != nullptr);  // IsQDQPairSupported should have checked that this exists.

  auto dq1_zp_type = dq1_zp_tensor_proto->data_type();

  if (dq1_zp_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    return RecomputeOuterQDQZeroPointAndScale<uint8_t>(graph, *q1, *dq1, *q2, *dq2);
  }

  if (dq1_zp_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return RecomputeOuterQDQZeroPointAndScale<int8_t>(graph, *q1, *dq1, *q2, *dq2);
  }

  if (dq1_zp_type == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    return RecomputeOuterQDQZeroPointAndScale<uint16_t>(graph, *q1, *dq1, *q2, *dq2);
  }

  if (dq1_zp_type == ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    return RecomputeOuterQDQZeroPointAndScale<int16_t>(graph, *q1, *dq1, *q2, *dq2);
  }

  return false;  // Unsupported zero-point type
}

Status DoubleQDQPairsRemover::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  const GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (const auto& dq1_index : node_topology_list) {
    NodeIndex q1_index = 0;
    NodeIndex q2_index = 0;
    NodeIndex dq2_index = 0;
    if (IsReducibleDoubleQDQSequence(graph, q1_index, dq1_index, q2_index, dq2_index)) {
      graph.RemoveEdge(q1_index, dq1_index, 0, 0);
      graph.RemoveEdge(dq1_index, q2_index, 0, 0);
      graph.RemoveEdge(q2_index, dq2_index, 0, 0);
      graph_utils::ReplaceNodeInput(*graph.GetNode(dq2_index), 0, *graph.GetNode(dq1_index)->MutableInputDefs()[0]);
      graph.AddEdge(q1_index, dq2_index, 0, 0);
      graph.RemoveNode(q2_index);
      graph.RemoveNode(dq1_index);
      modified = true;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
