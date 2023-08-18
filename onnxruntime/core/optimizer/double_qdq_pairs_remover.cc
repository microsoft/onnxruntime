// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/double_qdq_pairs_remover.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include <cassert>

namespace onnxruntime {

Status DoubleQDQPairsRemover::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  const GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (const auto& self_index : node_topology_list) {
    NodeIndex parent_index = 0;
    NodeIndex child_index = 0;
    NodeIndex grandchild_index = 0;
    if (IsNodeRemovable(graph, self_index, parent_index, child_index, grandchild_index)) {
      graph.RemoveEdge(parent_index, self_index, 0, 0);
      graph.RemoveEdge(self_index, child_index, 0, 0);
      graph.RemoveEdge(child_index, grandchild_index, 0, 0);
      graph_utils::ReplaceNodeInput(*graph.GetNode(grandchild_index), 0, *graph.GetNode(self_index)->MutableInputDefs()[0]);
      graph.AddEdge(parent_index, grandchild_index, 0, 0);
      graph.RemoveNode(child_index);
      graph.RemoveNode(self_index);
      modified = true;
    }
  }
  return Status::OK();
}

bool DoubleQDQPairsRemover::IsNodeRemovable(
    Graph& graph,
    const NodeIndex& self_index,
    NodeIndex& parent_index,
    NodeIndex& child_index,
    NodeIndex& grandchild_index) {
  // Check if the self is a DQ, and have one parent and one child, and cannot be a graph output
  Node* self = graph.GetNode(self_index);
  if (self == nullptr ||
      self->OpType() != "DequantizeLinear" ||
      self->GetInputEdgesCount() != 1 ||
      self->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*self)) {
    return false;
  }

  // child should be a Q, and have only one child, and cannot be a graph output
  child_index = self->OutputEdgesBegin()->GetNode().Index();
  const Node* child = graph.GetNode(child_index);
  if (child == nullptr ||
      child->OpType() != "QuantizeLinear" ||
      child->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*child)) {
    return false;
  }

  // parent should be a Q, and have only one output, and cannot be a graph output
  parent_index = self->InputEdgesBegin()->GetNode().Index();
  Node* parent = graph.GetNode(parent_index);
  if (parent == nullptr ||
      parent->GetOutputEdgesCount() != 1 ||
      parent->OpType() != "QuantizeLinear" ||
      graph.NodeProducesGraphOutput(*parent)) {
    return false;
  }

  // grandchild should be a DQ
  grandchild_index = child->OutputEdgesBegin()->GetNode().Index();
  Node* grandchild = graph.GetNode(grandchild_index);
  if (grandchild == nullptr ||
      grandchild->OpType() != "DequantizeLinear") {
    return false;
  }

  const auto get_constant_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  // Each QDQ pair (i.e., parent -> self, child -> grandchild) has to meet the following additional requirements:
  // - Scalar/constant zero-point and scale.
  // - Zero-point input must exist.
  // - The DQ and Q ops within a pair must have the same scale and zero-point.
  //   However, each pair is allowed to have different scales and zero-points.
  if (!QDQ::IsQDQPairSupported(*parent, *self, get_constant_initializer, graph.ModelPath()) ||
      !QDQ::IsQDQPairSupported(*child, *grandchild, get_constant_initializer, graph.ModelPath())) {
    return false;
  }

  const auto& self_input_defs = self->InputDefs();
  const ONNX_NAMESPACE::TensorProto* self_zp_tensor_proto = graph.GetConstantInitializer(
      self_input_defs[InputIndex::ZERO_POINT_ID]->Name(), true);

  assert(self_zp_tensor_proto != nullptr);  // IsQDQPairSupported should have checked that this exists.

  auto self_zp_type = self_zp_tensor_proto->data_type();

  // The two QDQ pairs may have different zero-points and scales. Ex: Q1 -> DQ1 -> Q2 -> DQ2, where
  // the first pair has (zp1, scale1) and the second pair has (zp2, scale2).
  // After removing the middle two nodes, the zero point and scale of the final (outer) ops must be recomputed.
  if (self_zp_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
    return ResetParentAndGrandchildZeroPointAndScale<uint8_t>(graph, *self, *child, *parent, *grandchild);
  }

  if (self_zp_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    return ResetParentAndGrandchildZeroPointAndScale<int8_t>(graph, *self, *child, *parent, *grandchild);
  }

  return false;  // Unsupported zero-point type
}

template <typename T>
bool DoubleQDQPairsRemover::FindNewZeroPointAndScale(const Graph& graph, const Node& node1, const Node& node2,
                                                     float& new_scale, T& new_zero_point, bool& skip_reset) {
  // scale & zero point share same initializer, no need to reset the value
  const std::string& node1_scale_name = node1.InputDefs()[InputIndex::SCALE_ID]->Name();
  const std::string& node2_scale_name = node2.InputDefs()[InputIndex::SCALE_ID]->Name();
  const std::string& node1_zp_name = node1.InputDefs()[InputIndex::ZERO_POINT_ID]->Name();
  const std::string& node2_zp_name = node2.InputDefs()[InputIndex::ZERO_POINT_ID]->Name();
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

template <typename T>
void DoubleQDQPairsRemover::ApplyNewInputValue(Graph& graph, Node& node, const InputIndex& index, T value) {
  const auto* input_tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[index]->Name());
  Initializer input_init{*input_tensor, graph.ModelPath()};
  TensorProto new_input_tensor(*input_tensor);
  input_init.data<T>()[0] = value;
  input_init.ToProto(new_input_tensor);
  auto new_name = graph.GenerateNodeArgName("DoubleQDQRemoved_" + node.InputDefs()[index]->Name());
  new_input_tensor.set_name(new_name);
  NodeArg& new_input = graph_utils::AddInitializer(graph, new_input_tensor);
  graph_utils::ReplaceNodeInput(node, index, new_input);
}

template <typename ZeroPointType>
bool DoubleQDQPairsRemover::ResetParentAndGrandchildZeroPointAndScale(Graph& graph, const Node& self,
                                                                      const Node& child, Node& parent,
                                                                      Node& grandchild) {
  bool skip_reset = false;
  float new_scale = 0.0f;
  ZeroPointType new_zero_point = 0;
  if (!FindNewZeroPointAndScale(graph, self, child, new_scale, new_zero_point, skip_reset)) {
    return false;
  }
  if (skip_reset) {
    return true;
  }
  ApplyNewInputValue(graph, grandchild, InputIndex::SCALE_ID, new_scale);
  ApplyNewInputValue(graph, parent, InputIndex::SCALE_ID, new_scale);
  ApplyNewInputValue(graph, grandchild, InputIndex::ZERO_POINT_ID, new_zero_point);
  ApplyNewInputValue(graph, parent, InputIndex::ZERO_POINT_ID, new_zero_point);

  return true;
}
}  // namespace onnxruntime
