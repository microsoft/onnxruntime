#include "core/providers/qnn/builder/qnn_conv_activation_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

static const NodeUnit* GetOnlyChildOfType(const GraphViewer& graph_viewer,
                                          const NodeUnit& parent_node_unit,
                                          gsl::span<const std::string_view> child_op_types,
                                          const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                          const std::unordered_set<const NodeUnit*>& handled_node_units) {
  const Node& parent_node = parent_node_unit.GetNode();

  // Parent must have a single child (1 output edge) and must not produce a graph output.
  if (parent_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(parent_node)) {
    return nullptr;
  }

  // Child must be of a valid type.
  const Node& child_node = parent_node.OutputEdgesBegin()->GetNode();
  const std::string& child_type = child_node.OpType();
  bool is_valid_child_type = false;

  for (const auto& valid_op_type : child_op_types) {
    if (valid_op_type == child_type) {
      is_valid_child_type = true;
      break;
    }
  }

  if (!is_valid_child_type) {
    return nullptr;
  }

  const auto child_node_unit_it = node_unit_map.find(&child_node);
  assert(child_node_unit_it != node_unit_map.end());
  const NodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (handled_node_units.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

static bool CanClipBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& clip_node_unit,
                             const NodeUnit& q_node_unit) {
  assert(clip_node_unit.OpType() == "Clip" && q_node_unit.OpType() == QDQ::QOpName);
  // TODO(adrianlizarraga): Implement.
  (void)qnn_model_wrapper;
  return true;
}

static bool CanReluBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& relu_node_unit,
                             const NodeUnit& q_node_unit) {
  assert(relu_node_unit.OpType() == "Relu" && q_node_unit.OpType() == QDQ::QOpName);
  const auto& q_inputs = q_node_unit.GetNode().InputDefs();

  // Require an explicit zero-point input for now.
  if (q_inputs.size() != 3 || !q_inputs[QDQ::ZERO_POINT_ID]->Exists()) {
    return false;
  }

  std::vector<int32_t> zero_points;
  int32_t zp_data_type = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
  Status status = qnn_model_wrapper.UnpackZeroPoints(q_inputs[QDQ::ZERO_POINT_ID]->Name(),
                                                     zero_points, zp_data_type);

  // Should only have one zero-point (per-tensor).
  if (!status.IsOK() || zero_points.size() != 1) {
    return false;
  }

  int32_t onnx_zero_point = -zero_points[0];  // QNN zero-points are negated.

  // Relu is redundant if the zero-point is set to the smallest quantized value.
  switch (zp_data_type) {
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT8:
      return onnx_zero_point == static_cast<int32_t>(std::numeric_limits<int8_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT8:
      return onnx_zero_point == static_cast<int32_t>(std::numeric_limits<uint8_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT16:
      return onnx_zero_point == static_cast<int32_t>(std::numeric_limits<int16_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT16:
      return onnx_zero_point == static_cast<int32_t>(std::numeric_limits<uint16_t>::lowest());
    default:
      return false;
  }
}

static bool CanActivationBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& activation_node_unit,
                                   const NodeUnit& q_node_unit) {
  const std::string& activation_type = activation_node_unit.OpType();

  if (activation_type == "Relu") {
    return CanReluBeRemoved(qnn_model_wrapper, activation_node_unit, q_node_unit);
  }

  if (activation_type == "Clip") {
    return CanClipBeRemoved(qnn_model_wrapper, activation_node_unit, q_node_unit);
  }

  return false;
}

// adjust for an optional input/output that has an entry but does not exist
static int NumActualValues(const Node& node, bool input) {
  const auto& defs = input ? node.InputDefs() : node.OutputDefs();
  return gsl::narrow_cast<int>(std::count_if(defs.cbegin(), defs.cend(),
                                             [](const NodeArg* def) { return def && def->Exists(); }));
}

static std::vector<const Node*> FindQDQNodes(const GraphViewer& graph_viewer, const Node& node, bool find_dq_nodes) {
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

static std::optional<QDQ::NodeGroup> GetConvQDQNodeGroup(
    const GraphViewer& graph_viewer,
    const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
    const std::unordered_set<const NodeUnit*>& handled_node_units,
    const Node& conv_node,
    const Node& q_node) {
  assert((conv_node.OpType() == "Conv" || conv_node.OpType() == "ConvTranspose") &&
         q_node.OpType() == QDQ::QOpName);
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, conv_node, /*find_dq_nodes*/ true);
  std::vector<const Node*> q_nodes = {&q_node};
  int num_dq_inputs = NumActualValues(conv_node, /*input*/ true);

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return std::nullopt;
  }

  for (const auto* dq_node : dq_nodes) {
    if (graph_viewer.NodeProducesGraphOutput(*dq_node)) {
      return std::nullopt;
    }

    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == conv_node.Index();
    if (!dq_has_single_output_edge_to_target) {
      return std::nullopt;
    }

    const auto dq_node_unit_it = node_unit_map.find(dq_node);
    assert(dq_node_unit_it != node_unit_map.end());
    const NodeUnit* dq_node_unit = dq_node_unit_it->second;

    if (handled_node_units.count(dq_node_unit) != 0) {
      return std::nullopt;
    }

    if (dq_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return std::nullopt;
    }
  }

  // input and output types need to be same
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return std::nullopt;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (dt_weight != dt_input) {
      return std::nullopt;
    }
  }

  if (dq_nodes.size() == 3) {  // has bias
    int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
      return std::nullopt;
    }
  }

  QDQ::NodeGroup node_group;
  node_group.dq_nodes.reserve(dq_nodes.size());
  node_group.q_nodes.reserve(q_nodes.size());
  node_group.target_node = conv_node.Index();
  auto get_node_idx = [&](const Node* n) { return n->Index(); };
  std::transform(dq_nodes.begin(), dq_nodes.end(), std::back_inserter(node_group.dq_nodes), get_node_idx);
  std::transform(q_nodes.begin(), q_nodes.end(), std::back_inserter(node_group.q_nodes), get_node_idx);
  return node_group;
}

Status TryConvActivationFusion(/*out*/ std::vector<const NodeUnit*>& fused_nodes,
                               QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& conv_node_unit,
                               const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                               const std::unordered_set<const NodeUnit*>& handled_node_units,
                               const logging::Logger& logger,
                               bool do_op_validation) {
  // Expect that this function is called with a standalone Conv or ConvTranspose.
  assert((conv_node_unit.OpType() == "Conv" || conv_node_unit.OpType() == "ConvTranspose") &&
         conv_node_unit.UnitType() == NodeUnit::Type::SingleNode);

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Conv must have a single Relu or Clip child.
  const std::array<std::string_view, 2> activation_op_types = {"Relu", "Clip"};
  const NodeUnit* activation_node_unit = GetOnlyChildOfType(graph_viewer, conv_node_unit, activation_op_types,
                                                            node_unit_map, handled_node_units);
  if (activation_node_unit == nullptr) {
    return Status::OK();
  }

  // Relu/Clip must have a single Q child.
  const std::array<std::string_view, 1> q_op_types = {QDQ::QOpName};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, *activation_node_unit, q_op_types,
                                                   node_unit_map, handled_node_units);

  if (q_node_unit == nullptr) {
    return Status::OK();
  }

  // Check if Clip/Relu can be removed because the Q node provides an equivalent effect.
  if (!CanActivationBeRemoved(qnn_model_wrapper, *activation_node_unit, *q_node_unit)) {
    return Status::OK();
  }

  // Create a QDQ node group with DQ* -> Conv -> Q
  const Node& conv_node = conv_node_unit.GetNode();
  const Node& activation_node = activation_node_unit->GetNode();
  const Node& q_node = q_node_unit->GetNode();
  std::optional<QDQ::NodeGroup> qdq_node_group = GetConvQDQNodeGroup(graph_viewer,
                                                                     node_unit_map,
                                                                     handled_node_units,
                                                                     conv_node,
                                                                     q_node);

  if (!qdq_node_group.has_value()) {
    return Status::OK();
  }

  NodeUnit qdq_node_unit(graph_viewer, *qdq_node_group);

  // Create a temporary QnnModelWrapper for validation only. We need to be sure that this fusion will work before
  // modifying the actual QnnModelWrapper. This allows us to revert to the traditional OpBuilder workflow if this
  // fusion doesn't work out.
  QnnModelWrapper tmp_model_wrapper(graph_viewer,
                                    logger,
                                    qnn_model_wrapper.GetQnnInterface(),
                                    qnn_model_wrapper.GetQnnBackendHandle(),
                                    qnn_model_wrapper.GetInputIndexMap(),
                                    qnn_model_wrapper.GetOutputIndexMap(),
                                    qnn_model_wrapper.GetInitializerLookup(),
                                    qnn_model_wrapper.GetQnnBackendType());

  const auto* conv_op_builder = qnn::GetOpBuilder(qdq_node_unit.OpType());
  if (conv_op_builder == nullptr) {
    return Status::OK();
  }

  QNN_RETURN_OK_IF_ERROR(conv_op_builder->IsOpSupported(tmp_model_wrapper, qdq_node_unit, logger), logger);

  // ====== The following statements modify qnn_model_wrapper. ========
  // Validation passed, so we're now committed to doing a fusion.
  // If we encounter an error, we return it directly to caller.
  LOGS(logger, VERBOSE) << " Adding Conv + Activation via fusion. conv_node name: [" << conv_node.Name()
                        << "] activation_node optype: [" << activation_node.OpType()
                        << "] activation_node name: [" << activation_node.Name()
                        << "]";

  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(conv_op_builder->IsOpSupported(qnn_model_wrapper, qdq_node_unit, logger));
  } else {
    ORT_RETURN_IF_ERROR(conv_op_builder->AddToModelBuilder(qnn_model_wrapper, qdq_node_unit, logger));
  }

  // Success. Add all nodes to fused_nodes so that caller can mark them as handled.
  for (const Node* dq_node : qdq_node_unit.GetDQNodes()) {
    const auto dq_node_unit_it = node_unit_map.find(dq_node);
    ORT_RETURN_IF(dq_node_unit_it == node_unit_map.end(), "DQ node does not have a NodeUnit");
    fused_nodes.push_back(dq_node_unit_it->second);
  }

  fused_nodes.push_back(&conv_node_unit);
  fused_nodes.push_back(activation_node_unit);
  fused_nodes.push_back(q_node_unit);

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
