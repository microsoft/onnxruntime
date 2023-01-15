// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/double_qdq_pairs_remover.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

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
      self->InputDefs().size() != InputIndex::TOTAL_COUNT ||
      graph.NodeProducesGraphOutput(*self)) {
    return false;
  }

  // Type is either "tensor(uint8)" or  "tensor(int8)"
  const auto self_zp_type = *self->InputDefs()[InputIndex::ZERO_POINT_ID]->Type();
  // child should be a Q, and have only one child, have the same type as self, and cannot be a graph output
  child_index = self->OutputEdgesBegin()->GetNode().Index();
  const Node* child = graph.GetNode(child_index);
  if (child == nullptr ||
      child->OpType() != "QuantizeLinear" ||
      child->GetOutputEdgesCount() != 1 ||
      child->InputDefs().size() != InputIndex::TOTAL_COUNT ||
      *child->InputDefs()[InputIndex::ZERO_POINT_ID]->Type() != self_zp_type ||
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
  if (!QDQ::IsQDQPairSupported(*parent, *self, get_constant_initializer, graph.ModelPath()) ||
      !QDQ::IsQDQPairSupported(*child, *grandchild, get_constant_initializer, graph.ModelPath())) {
    return false;
  }
  float new_scale = 0.0f;
  if (self_zp_type == "tensor(uint8)") {
    uint8_t new_zero_point = 0;
    if (!FindNewZeroPointAndScale(graph, *self, *child, new_scale, new_zero_point)) {
      return false;
    }
    ApplyNewInputValue(graph, *grandchild, InputIndex::SCALE_ID, new_scale);
    ApplyNewInputValue(graph, *parent, InputIndex::SCALE_ID, new_scale);
    ApplyNewInputValue(graph, *grandchild, InputIndex::ZERO_POINT_ID, new_zero_point);
    ApplyNewInputValue(graph, *parent, InputIndex::ZERO_POINT_ID, new_zero_point);
  } else {
    int8_t new_zero_point = 0;
    if (!FindNewZeroPointAndScale(graph, *self, *child, new_scale, new_zero_point)) {
      return false;
    }
    ApplyNewInputValue(graph, *grandchild, InputIndex::SCALE_ID, new_scale);
    ApplyNewInputValue(graph, *parent, InputIndex::SCALE_ID, new_scale);
    ApplyNewInputValue(graph, *grandchild, InputIndex::ZERO_POINT_ID, new_zero_point);
    ApplyNewInputValue(graph, *parent, InputIndex::ZERO_POINT_ID, new_zero_point);
  }
  return true;
}

template <typename T>
bool DoubleQDQPairsRemover::FindNewZeroPointAndScale(const Graph& graph, const Node& node1, const Node& node2, float& new_scale, T& new_zero_point) {
  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* node1_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node1.InputDefs()[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* node2_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node2.InputDefs()[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* node1_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node1.InputDefs()[InputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* node2_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, node2.InputDefs()[InputIndex::ZERO_POINT_ID]->Name());
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
}  // namespace onnxruntime
