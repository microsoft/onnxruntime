// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace QDQ {
namespace {

constexpr bool Is16BitIntType(int32_t data_type) {
  return (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16) ||
         (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16);
}

// adjust for an optional input/output that has an entry but does not exist
int NumActualValues(const Node& node, bool input) {
  const auto& defs = input ? node.InputDefs() : node.OutputDefs();
  return gsl::narrow_cast<int>(std::count_if(defs.cbegin(), defs.cend(),
                                             [](const NodeArg* def) { return def && def->Exists(); }));
}

std::vector<const Node*> FindQDQNodes(const GraphViewer& graph_viewer, const Node& node, bool find_dq_nodes) {
  // First get all the upstream (DQ) or downstream (Q) nodes
  std::vector<const Node*> nodes =
      find_dq_nodes ? graph_utils::FindParentsByType(node, QDQ::DQOpName)
                    : graph_utils::FindChildrenByType(node, QDQ::QOpName);

  // Remove all the nodes which are not in the graph_viewer
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [&graph_viewer](const Node* _node) {
                               return _node == nullptr || graph_viewer.GetNode(_node->Index()) == nullptr;
                             }),
              nodes.end());

  return nodes;
}
}  // namespace

bool NodeGroupSelector::CheckQDQNodes(const GraphViewer& graph_viewer, const Node& node,
                                      const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes,
                                      int num_dq_inputs,
                                      bool is_empty_q_nodes_allowed) const {
  if (num_dq_inputs == -1) {
    num_dq_inputs = NumActualValues(node, true);
  }

  // The input is a Graph Viewer, so cannot use graph_utils or optimizer_utils
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status = NodeGroup::CanCreateNodeGroup(graph_viewer, node, dq_nodes, q_nodes);
      !qdq_validation_status.IsOK()) {
    return false;
  }

  if (q_nodes.empty()) {
    return is_empty_q_nodes_allowed;
  }

  int num_outputs = NumActualValues(node, false);  // number of outputs that exist
  return (num_outputs == gsl::narrow_cast<int>(q_nodes.size())) &&
         q_nodes.size() == node.GetOutputEdgesCount() &&
         !graph_viewer.NodeProducesGraphOutput(node);
}

std::optional<NodeGroup> NodeGroupSelector::GetQDQSelection(const GraphViewer& graph_viewer, const Node& node) const {
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, node, true);
  std::vector<const Node*> q_nodes = FindQDQNodes(graph_viewer, node, false);
  if (!Check(graph_viewer, node, dq_nodes, q_nodes)) {
    return std::nullopt;
  }

  NodeGroup node_group;
  node_group.dq_nodes.reserve(dq_nodes.size());
  node_group.q_nodes.reserve(q_nodes.size());
  node_group.target_node = node.Index();
  auto get_node_idx = [&](const Node* n) { return n->Index(); };
  std::transform(dq_nodes.begin(), dq_nodes.end(), std::back_inserter(node_group.dq_nodes), get_node_idx);
  std::transform(q_nodes.begin(), q_nodes.end(), std::back_inserter(node_group.q_nodes), get_node_idx);
  return node_group;
}

std::optional<NodesToOptimizeIndices> BaseSelector::Select(const GraphViewer& graph_viewer, const Node& node) const {
  const std::string_view node_ep = node.GetExecutionProviderType();

  if (!compatible_providers_.empty() &&
      std::find(compatible_providers_.begin(), compatible_providers_.end(), node_ep) == compatible_providers_.end()) {
    return std::nullopt;
  }

  const auto qdq_group = node_group_selector_->GetQDQSelection(graph_viewer, node);
  if (!qdq_group.has_value()) {
    return std::nullopt;
  }

  NodesToOptimizeIndicesBuilder builder;
  // TODO(edgchen1) update NodeGroup to use InlinedVector
  builder.input_nodes.assign(qdq_group->dq_nodes.begin(), qdq_group->dq_nodes.end());
  builder.output_nodes.assign(qdq_group->q_nodes.begin(), qdq_group->q_nodes.end());
  // builder.input_nodes = qdq_group->dq_nodes;
  // builder.output_nodes = qdq_group->q_nodes;
  builder.target_node = qdq_group->target_node;

  UpdateBuilder(builder);
  return builder.Build();
}

bool DropQDQNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     const std::vector<const Node*>& dq_nodes,
                                     const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input != dt_output) {
    return false;
  }

  if (!allow_16bit_ && Is16BitIntType(dt_input)) {
    return false;
  }

  const Node& dq_node = *dq_nodes.front();
  const Node& q_node = *q_nodes.front();

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph_viewer.ModelPath());
}

bool DropDQNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  constexpr int num_dq_inputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status = NodeGroup::CanCreateNodeGroup(graph_viewer, node, dq_nodes, q_nodes);
      !qdq_validation_status.IsOK()) {
    return false;
  }

  (void)q_nodes;
  const Node& dq_node = *dq_nodes.front();
  const int32_t dt_input = dq_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && Is16BitIntType(dt_input)) {
    return false;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsDQSupported(dq_node, get_const_initializer);
}

bool UnaryNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input != dt_output) {
    return false;
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && Is16BitIntType(dt_input)) {
    return false;
  }

  return true;
}

bool BinaryNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // All input and output types must match.
  if (dt_input_1 != dt_input_2 || dt_input_1 != dt_output) {
    return false;
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && Is16BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

bool VariadicNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                      const Node& node,
                                      const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  // All DQs' inputs and Q's output should have same data type
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  for (size_t dq_idx = 1; dq_idx < dq_nodes.size(); dq_idx++) {
    if (dt_input != dq_nodes[dq_idx]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }
  }

  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  for (size_t q_idx = 1; q_idx < q_nodes.size(); q_idx++) {
    if (dt_output != q_nodes[q_idx]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }
  }

  if (dt_input != dt_output) {
    return false;
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && Is16BitIntType(dt_input)) {
    return false;
  }

  return true;
}

void InputVariadicSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.num_input_defs = 1;  // set to 1 as the first input is variadic
}

bool SplitNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  const Node& dq_node = *dq_nodes.front();
  int32_t dt_input = dq_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // All Q outputs should have same data type and (optionally) equal quantization parameters as the input.
  for (size_t q_idx = 0; q_idx < q_nodes.size(); q_idx++) {
    const Node& q_node = *q_nodes[q_idx];

    if (dt_input != q_node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }

    if (req_equal_quant_params_ &&
        !IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph_viewer.ModelPath())) {
      return false;
    }
  }

  return true;
}

void SplitSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.num_output_defs = 1;  // set to 1 as the first output is variadic
}

bool ConvNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  // input and output types need to be same
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return false;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  if (dq_nodes.size() == 3) {  // has bias
    int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && (Is16BitIntType(dt_input) || Is16BitIntType(dt_weight))) {
    return false;
  }

  return true;
}

void ConvSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.input_nodes.resize(3, NodesToOptimizeIndices::kEmptyNodeIndex);
}

bool MatMulNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (dq_nodes.size() != 2) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && (Is16BitIntType(dt_input) || Is16BitIntType(dt_weight))) {
    return false;
  }

  // potential match for QLinearMatMul or MatMulIntegerToFloat
  bool qlinear = !q_nodes.empty();

  if (qlinear) {
    // QLinearMatMul
    if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
      return false;
    }

    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input == dt_output;
  } else {
    // can be converted to MatMulIntegerToFloat if EP supports that.
    return matmulintegertofloat_allowed_;
  }
}

bool GemmNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes,
                     -1 /*num_dq_inputs*/, true /*is_empty_q_nodes_allowed*/)) {
    return false;
  }

  // input and output types need to be same
  int32_t dt_A = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_B = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_A == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (dt_A != dt_B) {  // if A is signed int, B must be signed int
      return false;
    }
  }

  if (!q_nodes.empty()) {
    int32_t dt_Y = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_A != dt_Y) {  // activation and output must be same type
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && (Is16BitIntType(dt_A) || Is16BitIntType(dt_B))) {
    return false;
  }

  if (dq_nodes.size() < 3) {  // no bias
    return true;
  }

  if (node.GetAttributes().at("beta").f() != 1.0) {  // beta needs to be 1.0
    return false;
  }

  int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
}

void GemmSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.input_nodes.resize(3, NodesToOptimizeIndices::kEmptyNodeIndex);
}

bool WhereNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  // Where has 1 boolean input and 2 dq inputs
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  const int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // All input and output types must match.
  if (dt_input_1 != dt_input_2 || dt_input_1 != dt_output) {
    return false;
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && Is16BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

bool PadNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                 const std::vector<const Node*>& dq_nodes,
                                 const std::vector<const Node*>& q_nodes) const {
  // Pad can have 1 or 2 dq input, the optional input constant_value can be quantized or non-quantized.
  // QNN supports data input quantized with constant_value input non-quantized.
  int num_dq_inputs = static_cast<int>(dq_nodes.size());
  if (num_dq_inputs > 2) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, num_dq_inputs)) {
    return false;
  }

  const int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dq_nodes.size() > 1) {
    const int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input_1 == dt_input_2 &&
           dt_input_1 == dt_output;
  } else {
    return dt_input_1 == dt_output;
  }
}

bool InstanceAndLayerNormalizationNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                                           const Node& node,
                                                           const std::vector<const Node*>& dq_nodes,
                                                           const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_bias = 0;
  bool has_bias = false;
  // bias is optional for LayerNorm
  if (dq_nodes.size() > 2) {
    has_bias = true;
    dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  }
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // Input, output, need to be the same type. The bias is int32.
  // Scale can be different with input for a16w8 case
  return (dt_input == dt_output) &&
         (has_bias ? dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32 : true);
}

bool BatchNormalizationNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                                const Node& node,
                                                const std::vector<const Node*>& dq_nodes,
                                                const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_scale = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return false;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_scale != dt_input) {
      return false;
    }
  }

  return true;
}

bool LogicalComparisonNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                               const Node& node,
                                               const std::vector<const Node*>& dq_nodes,
                                               const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, -1, true)) {
    return false;
  }

  int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input_1 == dt_input_2;
}

bool TopKNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  constexpr int num_dq_inputs = 1;
  constexpr int num_q_outputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status = QDQ::NodeGroup::CanCreateNodeGroup(graph_viewer, node, dq_nodes, q_nodes);
      !qdq_validation_status.IsOK()) {
    return false;
  }

  if (num_q_outputs != gsl::narrow_cast<int>(q_nodes.size())) {
    return false;
  }

  const Node& dq_node = *dq_nodes.front();
  const Node& q_node = *q_nodes.front();

  int32_t dt_input = dq_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input != dt_output) {
    return false;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph_viewer.ModelPath());
}

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
