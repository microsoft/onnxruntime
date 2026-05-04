// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
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

constexpr bool Is2BitIntType(int32_t data_type) {
  return (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT2) ||
         (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT2);
}

constexpr bool Is4BitIntType(int32_t data_type) {
  return (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4) ||
         (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4);
}

constexpr bool Is8BitIntType(int32_t data_type) {
  return (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
         (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8);
}

// Returns true if the data type is a sub-byte or byte quantized integer type
// suitable for MatMulNBits fusion (2, 4, or 8 bit).
constexpr bool IsNBitsIntType(int32_t data_type) {
  return Is2BitIntType(data_type) || Is4BitIntType(data_type) || Is8BitIntType(data_type);
}

// adjust for an optional input/output that has an entry but does not exist
int NumActualValues(const Node& node, bool input) {
  const auto& defs = input ? node.InputDefs() : node.OutputDefs();
  return gsl::narrow_cast<int>(
      std::count_if(defs.cbegin(), defs.cend(), [](const NodeArg* def) { return def && def->Exists(); }));
}

std::vector<const Node*> FindQDQNodes(const GraphViewer& graph_viewer, const Node& node, bool find_dq_nodes) {
  // First get all the upstream (DQ) or downstream (Q) nodes
  std::vector<const Node*> nodes = find_dq_nodes ? graph_utils::FindParentsByType(node, QDQ::DQOpName)
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
                                      const Node* redundant_clip_node, const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes, int num_dq_inputs,
                                      bool is_empty_q_nodes_allowed) const {
  if (num_dq_inputs == -1) {
    num_dq_inputs = NumActualValues(node, true);
  }

  // The input is a Graph Viewer, so cannot use graph_utils or optimizer_utils
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status =
          NodeGroup::CanCreateNodeGroup(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes);
      !qdq_validation_status.IsOK()) {
    return false;
  }

  if (q_nodes.empty()) {
    return is_empty_q_nodes_allowed;
  }

  int num_outputs = NumActualValues(node, false);  // number of outputs that exist
  return (num_outputs == gsl::narrow_cast<int>(q_nodes.size())) && q_nodes.size() == node.GetOutputEdgesCount() &&
         !graph_viewer.NodeProducesGraphOutput(node);
}

std::optional<NodeGroup> NodeGroupSelector::GetQDQSelection(const GraphViewer& graph_viewer, const Node& node) const {
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, node, true);

  // For redundant clip node, currently only support node with only one output, which is consumed by Clip/Relu->Q.
  const Node* clip_node = nullptr;
  if (node.GetOutputEdgesCount() == 1) {
    const Node& next_node = *node.OutputNodesBegin();
    if ((next_node.OpType() == "Relu" || next_node.OpType() == "Clip") && next_node.GetOutputEdgesCount() == 1 &&
        !graph_viewer.NodeProducesGraphOutput(next_node)) {
      clip_node = &next_node;
    }
  }

  std::vector<const Node*> q_nodes = FindQDQNodes(graph_viewer, (clip_node ? *clip_node : node), false);

  if (clip_node && (q_nodes.size() != 1 || !IsClipMadeRedundantByQ(graph_viewer.GetGraph(), *clip_node, *q_nodes[0]))) {
    return std::nullopt;
  }

  // When here, if clip_node is not nullptr, it is redundant.
  if (!Check(graph_viewer, node, clip_node, dq_nodes, q_nodes)) {
    return std::nullopt;
  }

  NodeGroup node_group;
  node_group.dq_nodes.reserve(dq_nodes.size());
  node_group.q_nodes.reserve(q_nodes.size());
  node_group.target_node = node.Index();
  if (clip_node) {
    node_group.redundant_clip_node = clip_node->Index();
  }
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

bool DropQDQNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                     const std::vector<const Node*>& dq_nodes,
                                     const std::vector<const Node*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, nullptr, dq_nodes, q_nodes, 1)) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  const Node& dq_node = *dq_nodes.front();
  const Node& q_node = *q_nodes.front();

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  if (!allow_nonpositive_scale_) {
    // IsQDQPairSupported will check that the scale is the same between q_node and dq_node.
    if (!IsQOrDQScalePositiveConstantScalar(graph_viewer.GetGraph(), q_node, get_const_initializer,
                                            graph_viewer.ModelPath())) {
      return false;
    }
  }

  return IsQDQPairSupported(graph_viewer.GetGraph(), q_node, dq_node, get_const_initializer, graph_viewer.ModelPath());
}

bool DropDQNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status = NodeGroup::CanCreateNodeGroup(graph_viewer, node, nullptr, dq_nodes, q_nodes);
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

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsDQSupported(dq_node, get_const_initializer);
}

bool UnaryNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

bool ClipNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  // Clip can have 1, 2, or 3 DQ inputs:
  // - 1 DQ: only data input is quantized
  // - 2 DQ: data and min or max are quantized
  // - 3 DQ: data, min, and max are all quantized
  const size_t num_dq_nodes = dq_nodes.size();
  if (num_dq_nodes < 1 || num_dq_nodes > 3) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, static_cast<int>(num_dq_nodes))) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

bool BinaryNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

bool VariadicNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                      const Node* redundant_clip_node, const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

void InputVariadicSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.num_input_defs = 1;  // set to 1 as the first input is variadic
}

bool SplitNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, nullptr, dq_nodes, q_nodes, 1)) {
    return false;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  const Node& dq_node = *dq_nodes.front();
  int32_t dt_input = dq_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  // All Q outputs should have same data type and (optionally) equal quantization parameters as the input.
  for (size_t q_idx = 0; q_idx < q_nodes.size(); q_idx++) {
    const Node& q_node = *q_nodes[q_idx];

    if (dt_input != q_node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }

    if (req_equal_quant_params_ &&
        !IsQDQPairSupported(graph_viewer.GetGraph(), q_node, dq_node, get_const_initializer, graph_viewer.ModelPath())) {
      return false;
    }
  }

  return true;
}

void SplitSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.num_output_defs = 1;  // set to 1 as the first output is variadic
}

bool ConvNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  // input and output types need to be same
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return false;
  }

  if (!allow_4bit_weight_ && Is4BitIntType(dt_weight)) {
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

bool EinsumNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node, const Node* redundant_clip_node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    int32_t dt_input = dq_nodes[i]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (!allow_int8_ && dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
      return false;
    }
    if (!allow_16bit_ && Is16BitIntType(dt_input)) {
      return false;
    }
    if (!allow_4bit_ && Is4BitIntType(dt_input)) {
      return false;
    }
  }
  if (!q_nodes.empty()) {
    int32_t dt_input0 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_input0 != dt_output) {
      return false;
    }
  }
  return true;
}

bool ReciprocalNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                        const Node& node, const Node* redundant_clip_node,
                                        const std::vector<const Node*>& dq_nodes,
                                        const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    int32_t dt_input = dq_nodes[i]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (!allow_int8_ && dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
      return false;
    }
    if (!allow_16bit_ && Is16BitIntType(dt_input)) {
      return false;
    }
    if (!allow_4bit_ && Is4BitIntType(dt_input)) {
      return false;
    }
  }
  if (!q_nodes.empty()) {
    int32_t dt_input0 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_input0 != dt_output) {
      return false;
    }
  }
  return true;
}

bool MatMulNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
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

  // 4-bit int types must be explicitly allowed.
  if (!allow_4bit_ && (Is4BitIntType(dt_input) || Is4BitIntType(dt_weight))) {
    return false;
  }

  // potential match for QLinearMatMul or MatMulIntegerToFloat
  bool qlinear = !q_nodes.empty();

  if (qlinear) {
    // QLinearMatMul
    if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
      return false;
    }

    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input == dt_output;
  } else {
    // can be converted to MatMulIntegerToFloat if EP supports that.
    return matmulintegertofloat_allowed_;
  }
}

// Validate that a DQ node has the correct structure for MatMulNBits fusion.
// Supports three quantization granularities:
// - Blockwise: axis=0, block_size >= 16 and power-of-2, scale/zp rank 2
// - Per-tensor: scale is scalar (rank 0), no block_size attribute
// - Per-channel (axis=1): scale is 1D with shape [N], weight is 2D [K,N], no block_size attribute
// In all cases: weight type is 2/4/8-bit int, scale type is float or float16,
// weight/scale/zp are constant initializers.
static bool ValidateDQForMatMulNBits(const Graph& graph, const Node& dq_node) {
  const auto* weight_arg = dq_node.InputDefs()[0];
  const auto* scale_arg = dq_node.InputDefs()[1];
  const auto* zero_point_arg = dq_node.InputDefs().size() == 3 ? dq_node.InputDefs()[2] : nullptr;
  int32_t dt_weight = weight_arg->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_scales = scale_arg->TypeAsProto()->tensor_type().elem_type();

  if (dt_scales != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT &&
      dt_scales != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
    return false;
  }

  if (!IsNBitsIntType(dt_weight)) {
    return false;
  }

  // weight, scale and zero points (if exists) must be constants
  const auto* weight_tensor_proto = graph.GetConstantInitializer(weight_arg->Name(), true);
  const auto* scale_tensor_proto = graph.GetConstantInitializer(scale_arg->Name(), true);
  const auto* zp_tensor_proto = zero_point_arg ? graph.GetConstantInitializer(zero_point_arg->Name(), true) : nullptr;

  if (!weight_tensor_proto || !scale_tensor_proto) {
    return false;
  }

  if (zero_point_arg && !zp_tensor_proto) {
    return false;
  }

  // weight must be rank 2
  if (weight_tensor_proto->dims_size() != 2) {
    return false;
  }

  const auto& dq_attrs = dq_node.GetAttributes();
  const auto block_size_iter = dq_attrs.find("block_size");
  const bool has_block_size = block_size_iter != dq_attrs.end() && block_size_iter->second.i() > 0;

  if (has_block_size) {
    // --- Blockwise path (existing logic) ---
    if (const auto a_iter = dq_attrs.find("axis"); a_iter == dq_attrs.end() || a_iter->second.i() != 0) {
      return false;
    }

    auto block_size = block_size_iter->second.i();
    if (block_size < 16 || ((block_size - 1) & block_size)) {
      return false;
    }

    if (scale_tensor_proto->dims_size() != 2 ||
        (zp_tensor_proto && zp_tensor_proto->dims_size() != 2)) {
      return false;
    }

    if ((weight_tensor_proto->dims()[0] + block_size - 1) / block_size != scale_tensor_proto->dims()[0] ||
        weight_tensor_proto->dims()[1] != scale_tensor_proto->dims()[1] ||
        (zp_tensor_proto && (zp_tensor_proto->dims()[0] != scale_tensor_proto->dims()[0] ||
                             zp_tensor_proto->dims()[1] != scale_tensor_proto->dims()[1]))) {
      return false;
    }
  } else {
    // --- Per-tensor or per-channel path ---
    int scale_rank = scale_tensor_proto->dims_size();
    auto N = weight_tensor_proto->dims()[1];

    if (scale_rank == 0) {
      // Per-tensor: scalar scale, optional scalar zp
      if (zp_tensor_proto && zp_tensor_proto->dims_size() != 0) {
        return false;
      }
    } else if (scale_rank == 1 && scale_tensor_proto->dims()[0] == N) {
      // Per-channel (axis=1): scale shape [N], axis must be 1
      const auto a_iter = dq_attrs.find("axis");
      // DQ default axis is 1, so absent axis is OK
      if (a_iter != dq_attrs.end() && a_iter->second.i() != 1) {
        return false;
      }
      if (zp_tensor_proto && (zp_tensor_proto->dims_size() != 1 || zp_tensor_proto->dims()[0] != N)) {
        return false;
      }
    } else {
      // Unsupported quantization granularity
      return false;
    }
  }

  return true;
}

// Validate Gemm attributes for DQ->MatMulNBits fusion.
// Gemm must be equivalent to MatMul: alpha=1, transA=0, transB=0.
// If bias exists, beta must be 1 and bias shape must be [N].
static bool ValidateGemmForDQMatMulNBits(const Graph& graph, const Node& gemm_node, const Node& weight_dq_node) {
  if (const auto* alpha_attr = graph_utils::GetNodeAttribute(gemm_node, "alpha");
      alpha_attr && std::abs(alpha_attr->f() - 1.0f) > 1e-6f)
    return false;
  if (const auto* trans_a = graph_utils::GetNodeAttribute(gemm_node, "transA");
      trans_a && trans_a->i() != 0)
    return false;
  if (const auto* trans_b = graph_utils::GetNodeAttribute(gemm_node, "transB");
      trans_b && trans_b->i() != 0)
    return false;

  const auto& inputs = gemm_node.InputDefs();
  if (inputs.size() > 2 && inputs[2] && inputs[2]->Exists()) {
    // Bias exists — beta must be 1.0
    if (const auto* beta_attr = graph_utils::GetNodeAttribute(gemm_node, "beta");
        beta_attr && std::abs(beta_attr->f() - 1.0f) > 1e-6f)
      return false;

    // Bias shape must be [N] where N = weight dim 1. Prefer reading N and
    // bias length from constant initializers when available, and fall back to
    // NodeArg::Shape().
    const auto* weight_arg = weight_dq_node.InputDefs()[0];
    const auto* weight_initializer = graph.GetConstantInitializer(weight_arg->Name(), true);
    int64_t N = -1;

    if (weight_initializer) {
      if (weight_initializer->dims_size() != 2) {
        return false;
      }
      N = weight_initializer->dims(1);
    } else {
      const auto* weight_shape = weight_arg->Shape();
      if (!weight_shape || weight_shape->dim_size() != 2 ||
          !utils::HasDimValue(weight_shape->dim(1))) {
        return false;
      }
      N = weight_shape->dim(1).dim_value();
    }

    const auto* bias_arg = inputs[2];
    const auto* bias_initializer = graph.GetConstantInitializer(bias_arg->Name(), true);

    if (bias_initializer) {
      if (bias_initializer->dims_size() != 1 ||
          bias_initializer->dims(0) != N) {
        return false;
      }
    } else {
      const auto* bias_shape = bias_arg->Shape();
      if (!bias_shape || bias_shape->dim_size() != 1 ||
          !utils::HasDimValue(bias_shape->dim(0)) ||
          bias_shape->dim(0).dim_value() != N) {
        return false;
      }
    }
  }

  return true;
}

bool DQMatMulNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                      const Node* redundant_clip_node, const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  // Should not have any Q nodes
  if (!q_nodes.empty()) {
    return false;
  }

  const auto& graph = graph_viewer.GetGraph();
  const bool is_gemm = node.OpType() == "Gemm";

  if (is_gemm) {
    // Gemm: accept 1 DQ (weight only) or 2 DQs (weight + bias).
    if (dq_nodes.size() < 1 || dq_nodes.size() > 2) {
      return false;
    }
  } else {
    // MatMul: exactly 1 DQ input
    if (dq_nodes.size() != 1) {
      return false;
    }
  }

  // Find the weight DQ node — the one feeding input 1 (B)
  const Node* weight_dq = nullptr;
  for (const auto* dq : dq_nodes) {
    if (node.InputDefs()[1] == dq->OutputDefs()[0]) {
      weight_dq = dq;
      break;
    }
  }

  if (!weight_dq) {
    return false;
  }

  // Weight DQ must have exactly 1 output edge and not be a graph output
  if (!optimizer_utils::CheckOutputEdges(graph, *weight_dq, 1)) {
    return false;
  }

  if (is_gemm) {
    // If there's a second DQ node (for bias), it must feed input 2
    if (dq_nodes.size() == 2) {
      const Node* bias_dq = (dq_nodes[0] == weight_dq) ? dq_nodes[1] : dq_nodes[0];
      if (node.InputDefs().size() <= 2 || !node.InputDefs()[2] ||
          node.InputDefs()[2] != bias_dq->OutputDefs()[0]) {
        return false;
      }
    }

    // Validate Gemm attributes (alpha=1, transA=0, transB=0, beta=1 if bias)
    if (!ValidateGemmForDQMatMulNBits(graph, node, *weight_dq)) {
      return false;
    }
  }

  return ValidateDQForMatMulNBits(graph, *weight_dq);
}

bool GemmNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, -1 /*num_dq_inputs*/,
                     true /*is_empty_q_nodes_allowed*/)) {
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

  if (!allow_4bit_ && (Is4BitIntType(dt_A) || Is4BitIntType(dt_B))) {
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

void DQMatMulToMatMulNBitsSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  // Keep only the weight DQ (first entry). If a Gemm has a bias DQ, it will be in
  // position 1 — trim it so RemoveNodes does not delete it. The bias DQ's output
  // is wired to MatMulNBits input 5 in ProcessNewNode.
  builder.input_nodes.resize(1);
}

bool WhereNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  // Where has 1 boolean input and 2 dq inputs
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
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

  if (!allow_4bit_ && Is4BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

bool PadNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                 const std::vector<const Node*>& dq_nodes,
                                 const std::vector<const Node*>& q_nodes) const {
  // Pad can have 1 or 2 dq input, the optional input constant_value can be quantized or non-quantized.
  // QNN supports data input quantized with constant_value input non-quantized.
  int num_dq_inputs = static_cast<int>(dq_nodes.size());
  if (num_dq_inputs > 2) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, num_dq_inputs)) {
    return false;
  }

  const int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dq_nodes.size() > 1) {
    const int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input_1 == dt_input_2 && dt_input_1 == dt_output;
  } else {
    return dt_input_1 == dt_output;
  }
}

bool InstanceAndLayerNormalizationNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                                           const Node* redundant_clip_node,
                                                           const std::vector<const Node*>& dq_nodes,
                                                           const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
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

bool BatchNormalizationNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                                const Node* redundant_clip_node,
                                                const std::vector<const Node*>& dq_nodes,
                                                const std::vector<const Node*>& q_nodes) const {
  // BatchNormalization has 5 inputs: x, scale, bias, mean, var.
  // Require DQ on x and scale (indices 0,1). mean, var may optionally have DQ.
  const int num_dq_nodes = gsl::narrow_cast<int>(dq_nodes.size());
  if (num_dq_nodes < 3 || num_dq_nodes > 5) {
    return false;
  }

  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, num_dq_nodes)) {
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

bool LogicalComparisonNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                               const Node* redundant_clip_node, const std::vector<const Node*>& dq_nodes,
                                               const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, -1, true)) {
    return false;
  }

  int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input_1 == dt_input_2;
}

bool TopKNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  // Not support for now. Need to handle the indices output if we want to support it.
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  constexpr int num_q_outputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (const auto qdq_validation_status =
          QDQ::NodeGroup::CanCreateNodeGroup(graph_viewer, node, nullptr, dq_nodes, q_nodes);
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

  return IsQDQPairSupported(graph_viewer.GetGraph(), q_node, dq_node, get_const_initializer, graph_viewer.ModelPath());
}

bool CumSumNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  // Only the first input has DQ node
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input != dt_output) {
    return false;
  }

  return true;
}

bool ScatterElementsNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node, const Node* redundant_clip_node,
                                             const std::vector<const Node*>& dq_nodes,
                                             const std::vector<const Node*>& q_nodes) const {
  // ScatterElements has 1 INT32 input and 2 dq inputs
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }
  const int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // All input and output types must match.
  if (dt_input_1 != dt_input_2 || dt_input_1 != dt_output) {
    return false;
  }

  return true;
}

bool RMSNormalizationNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                              const Node* redundant_clip_node,
                                              const std::vector<const Node*>& dq_nodes,
                                              const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  // input and output need to be the same type.
  return (dt_input == dt_output);
}

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
