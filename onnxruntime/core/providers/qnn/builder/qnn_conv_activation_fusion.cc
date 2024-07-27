#include "core/providers/qnn/builder/qnn_conv_activation_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/framework/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

static const NodeUnit* GetOnlyChildOfType(const GraphViewer& graph_viewer,
                                          const NodeUnit& parent_node_unit,
                                          gsl::span<const std::string_view> child_op_types,
                                          const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                          const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group) {
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
  if (node_unit_to_qnn_node_group.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

static bool GetQScalarScaleZeroPoint(const QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& q_node_unit,
                                     /*out*/ float& scale,
                                     /*out*/ int32_t& zero_point,
                                     /*out*/ int32_t& zp_data_type) {
  assert(q_node_unit.OpType() == QDQ::QOpName);
  const auto& q_inputs = q_node_unit.GetNode().InputDefs();

  // Require an explicit zero-point input for now.
  if (q_inputs.size() != 3 || !q_inputs[QDQ::ZERO_POINT_ID]->Exists()) {
    return false;
  }

  std::vector<int32_t> zero_points;
  Status status = qnn_model_wrapper.UnpackZeroPoints(q_inputs[QDQ::ZERO_POINT_ID]->Name(),
                                                     zero_points, zp_data_type);

  // Should only have one zero-point (per-tensor).
  if (!status.IsOK() || zero_points.size() != 1) {
    return false;
  }
  zero_point = -zero_points[0];  // QNN zero-points are negated.

  std::vector<float> scales;
  status = qnn_model_wrapper.UnpackScales(q_inputs[QDQ::SCALE_ID]->Name(), scales);

  // Should only have one scale (per-tensor).
  if (!status.IsOK() || scales.size() != 1) {
    return false;
  }

  scale = scales[0];
  return true;
}

static bool GetQRminRmax(const QnnModelWrapper& qnn_model_wrapper,
                         const NodeUnit& q_node_unit,
                         /*out*/ float& rmin,
                         /*out*/ float& rmax) {
  int32_t zp_data_type = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
  int32_t zero_point = 0;
  float scale = 0.0f;

  if (!GetQScalarScaleZeroPoint(qnn_model_wrapper, q_node_unit, scale, zero_point, zp_data_type)) {
    return false;
  }

  switch (zp_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      rmin = scale * (std::numeric_limits<int8_t>::lowest() - zero_point);
      rmax = scale * (std::numeric_limits<int8_t>::max() - zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      rmin = scale * (std::numeric_limits<uint8_t>::lowest() - zero_point);
      rmax = scale * (std::numeric_limits<uint8_t>::max() - zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      rmin = scale * (std::numeric_limits<int16_t>::lowest() - zero_point);
      rmax = scale * (std::numeric_limits<int16_t>::max() - zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      rmin = scale * (std::numeric_limits<uint16_t>::lowest() - zero_point);
      rmax = scale * (std::numeric_limits<uint16_t>::max() - zero_point);
      break;
    }
    default:
      return false;
  }

  return true;
}

static bool GetClipMinMax(const QnnModelWrapper& qnn_model_wrapper,
                          const NodeUnit& clip_node_unit,
                          /*out*/ float& clip_min,
                          /*out*/ float& clip_max) {
  clip_min = std::numeric_limits<float>::lowest();
  clip_max = std::numeric_limits<float>::max();

  // Clip's min and max are attributes before opset 11.
  if (clip_node_unit.GetNode().SinceVersion() < 11) {
    NodeAttrHelper attr_helper(clip_node_unit);
    std::optional<float> min_opt = attr_helper.GetFloat("min");
    std::optional<float> max_opt = attr_helper.GetFloat("max");

    if (min_opt.has_value()) {
      clip_min = min_opt.value();
    }

    if (max_opt.has_value()) {
      clip_max = max_opt.value();
    }

    return true;
  }

  // After opset 11, min and max are inputs.
  const auto& inputs = clip_node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  auto get_min_or_max = [&qnn_model_wrapper](const NodeUnitIODef& input, /*out*/ float& result) -> bool {
    TensorInfo input_info = {};
    std::vector<uint8_t> raw_bytes;
    if (Status status = qnn_model_wrapper.GetTensorInfo(input, input_info); !status.IsOK()) {
      return false;
    }
    if (!input_info.is_initializer) {
      return false;
    }
    if (Status status = qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, raw_bytes);
        !status.IsOK()) {
      return false;
    }
    if (input_info.qnn_data_type != QNN_DATATYPE_FLOAT_32) {
      return false;
    }
    result = static_cast<float>(*reinterpret_cast<const float*>(raw_bytes.data()));
    return true;
  };

  if (num_inputs > 1 && inputs[1].node_arg.Exists()) {
    if (!get_min_or_max(inputs[1], clip_min)) {
      return false;
    }
  }

  if (num_inputs > 2 && inputs[2].node_arg.Exists()) {
    if (!get_min_or_max(inputs[2], clip_max)) {
      return false;
    }
  }

  return true;
}

static bool CanClipBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& clip_node_unit,
                             const NodeUnit& q_node_unit) {
  assert(clip_node_unit.OpType() == "Clip" && q_node_unit.OpType() == QDQ::QOpName);
  float rmin = 0.0f;
  float rmax = 0.0f;

  if (!GetQRminRmax(qnn_model_wrapper, q_node_unit, rmin, rmax)) {
    return false;
  }

  float clip_min = std::numeric_limits<float>::lowest();
  float clip_max = std::numeric_limits<float>::max();

  if (!GetClipMinMax(qnn_model_wrapper, clip_node_unit, clip_min, clip_max)) {
    return false;
  }

  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  if ((epsilon < clip_min - rmin) || (epsilon < rmax - clip_max)) {
    return false;
  }

  return true;
}

static bool CanReluBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& relu_node_unit,
                             const NodeUnit& q_node_unit) {
  assert(relu_node_unit.OpType() == "Relu" && q_node_unit.OpType() == QDQ::QOpName);
  int32_t zp_data_type = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
  int32_t zero_point = 0;
  float scale = 0.0f;

  if (!GetQScalarScaleZeroPoint(qnn_model_wrapper, q_node_unit, scale, zero_point, zp_data_type)) {
    return false;
  }

  // Relu is redundant if the zero-point is set to the smallest quantized value.
  switch (zp_data_type) {
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT8:
      return zero_point == static_cast<int32_t>(std::numeric_limits<int8_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT8:
      return zero_point == static_cast<int32_t>(std::numeric_limits<uint8_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT16:
      return zero_point == static_cast<int32_t>(std::numeric_limits<int16_t>::lowest());
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT16:
      return zero_point == static_cast<int32_t>(std::numeric_limits<uint16_t>::lowest());
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

static Status GetConvDQNodeUnits(
    /*out*/ std::vector<const NodeUnit*>& dq_node_units,
    const GraphViewer& graph_viewer,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const Node& conv_node,
    const Node& q_node) {
  assert((conv_node.OpType() == "Conv" || conv_node.OpType() == "ConvTranspose") &&
         q_node.OpType() == QDQ::QOpName);
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, conv_node, /*find_dq_nodes*/ true);
  std::vector<const Node*> q_nodes = {&q_node};
  int num_dq_inputs = NumActualValues(conv_node, /*input*/ true);

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  ORT_RETURN_IF_NOT(num_dq_inputs == static_cast<int>(dq_nodes.size()),
                    "Conv should be the only consumer of each DQ");

  for (const auto* dq_node : dq_nodes) {
    ORT_RETURN_IF(graph_viewer.NodeProducesGraphOutput(*dq_node),
                  "QDQ ", conv_node.OpType(), "'s input DQ node must not produce a graph output");

    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == conv_node.Index();
    ORT_RETURN_IF_NOT(dq_has_single_output_edge_to_target, "DQ should have a single output to Conv");
  }

  // input and output types need to be same
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(dt_input != dt_output, "Conv input[0] and output quantization types must match");

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    ORT_RETURN_IF(dt_weight != dt_input,
                  conv_node.OpType(), "'s input[0] and input[1] quantization types must match if input[0] is int8");
  }

  if (dq_nodes.size() == 3) {  // has bias
    int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    ORT_RETURN_IF(dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                  "QDQ ", conv_node.OpType(), " must have int32 quantized bias");
  }

  dq_node_units.reserve(dq_nodes.size());
  for (const auto* dq_node : dq_nodes) {
    const auto it = node_to_node_unit.find(dq_node);
    assert(it != node_to_node_unit.end());
    const NodeUnit* dq_node_unit = it->second;

    ORT_RETURN_IF_NOT(node_unit_to_qnn_node_group.count(dq_node_unit) == 0,
                      "DQ NodeUnit ", dq_node_unit->Name(), " has already been added to another QnnNodeGroup");
    ORT_RETURN_IF_NOT(dq_node_unit->UnitType() == NodeUnit::Type::SingleNode,
                      "Expect DQ to be a NodeUnit of type SingleNode");
    dq_node_units.push_back(dq_node_unit);
  }

  return Status::OK();
}

Status QnnConvActivationFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                                  gsl::span<const NodeUnit*> dq_node_units,
                                  const NodeUnit* conv_node_unit,
                                  const NodeUnit* activation_node_unit,
                                  const NodeUnit* q_node_unit,
                                  const logging::Logger& logger,
                                  bool validate) {
  QDQ::NodeGroup custom_node_group;
  custom_node_group.dq_nodes.reserve(dq_node_units.size());
  custom_node_group.q_nodes = std::vector<NodeIndex>{q_node_unit->Index()};
  custom_node_group.target_node = conv_node_unit->Index();
  auto get_node_idx = [](const NodeUnit* n) { return n->Index(); };
  std::transform(dq_node_units.begin(), dq_node_units.end(), std::back_inserter(custom_node_group.dq_nodes),
                 get_node_idx);

  NodeUnit custom_node_unit(qnn_model_wrapper.GetGraphViewer(), custom_node_group, activation_node_unit->GetNode());
  const auto* conv_op_builder = qnn::GetOpBuilder(custom_node_unit.OpType());
  if (conv_op_builder == nullptr) {
    return Status::OK();
  }

  if (validate) {
    return conv_op_builder->IsOpSupported(qnn_model_wrapper, custom_node_unit, logger);
  }

  return conv_op_builder->AddToModelBuilder(qnn_model_wrapper, custom_node_unit, logger, validate);
}

Status TryConvActivationFusion(/*out*/ std::optional<QnnNodeGroup>& qnn_node_group,
                               QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& conv_node_unit,
                               const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                               const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
                               const logging::Logger& logger) {
  // Expect that this function is called with a standalone Conv or ConvTranspose.
  assert((conv_node_unit.OpType() == "Conv" || conv_node_unit.OpType() == "ConvTranspose") &&
         conv_node_unit.UnitType() == NodeUnit::Type::SingleNode);

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Conv must have a single Relu or Clip child.
  const std::array<std::string_view, 2> activation_op_types = {"Relu", "Clip"};
  const NodeUnit* activation_node_unit = GetOnlyChildOfType(graph_viewer, conv_node_unit, activation_op_types,
                                                            node_to_node_unit, node_unit_to_qnn_node_group);
  if (activation_node_unit == nullptr) {
    return Status::OK();
  }

  // Relu/Clip must have a single Q child.
  const std::array<std::string_view, 1> q_op_types = {QDQ::QOpName};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, *activation_node_unit, q_op_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);

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
  std::vector<const NodeUnit*> dq_node_units;
  QNN_RETURN_OK_IF_ERROR(GetConvDQNodeUnits(dq_node_units,
                                            graph_viewer,
                                            node_to_node_unit,
                                            node_unit_to_qnn_node_group,
                                            conv_node,
                                            q_node),
                         logger);

  assert(dq_node_units.size() == 3 || dq_node_units.size() == 2);

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

  QNN_RETURN_OK_IF_ERROR(QnnConvActivationFusionAdd(tmp_model_wrapper,
                                                    dq_node_units,
                                                    &conv_node_unit,
                                                    activation_node_unit,
                                                    q_node_unit,
                                                    logger,
                                                    /*validate*/ true),
                         logger);

  // Validation passed, so create a QnnNodeGroup.
  // If we encounter an error, we return it directly to caller.
  LOGS(logger, VERBOSE) << "Will use Conv + Activation via fusion. conv_node name: [" << conv_node.Name()
                        << "] activation_node optype: [" << activation_node.OpType()
                        << "] activation_node name: [" << activation_node.Name()
                        << "]";

  qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::ConvActivationFusion;
  qnn_node_group->node_units_ = std::move(dq_node_units);
  qnn_node_group->node_units_.push_back(&conv_node_unit);
  qnn_node_group->node_units_.push_back(activation_node_unit);
  qnn_node_group->node_units_.push_back(q_node_unit);

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
