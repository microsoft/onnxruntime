#include "core/providers/qnn/builder/qnn_node_group/conv_activation_fusion.h"

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
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

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

static bool CanQRelaceRelu(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& q_node_unit) {
  assert(q_node_unit.OpType() == QDQ::QOpName);
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
    return CanQRelaceRelu(qnn_model_wrapper, q_node_unit);
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

static std::vector<const NodeUnit*> GetConvDQs(
    const GraphViewer& graph_viewer,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const Node& conv_node) {
  assert(conv_node.OpType() == "Conv" || conv_node.OpType() == "ConvTranspose");
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, conv_node, /*find_dq_nodes*/ true);
  int num_dq_inputs = NumActualValues(conv_node, /*input*/ true);

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return {};
  }

  std::vector<const NodeUnit*> dq_node_units;
  for (const auto* dq_node : dq_nodes) {
    if (graph_viewer.NodeProducesGraphOutput(*dq_node)) {
      return {};
    }

    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == conv_node.Index();
    if (!dq_has_single_output_edge_to_target) {
      return {};
    }

    const auto it = node_to_node_unit.find(dq_node);
    if (it == node_to_node_unit.end()) {
      return {};
    }

    const NodeUnit* dq_node_unit = it->second;

    if (!dq_node_unit || node_unit_to_qnn_node_group.count(dq_node_unit) != 0) {
      return {};
    }

    if (dq_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return {};
    }

    dq_node_units.push_back(dq_node_unit);
  }

  return dq_node_units;
}

static bool IsValidQDQConv(gsl::span<const NodeUnit*> dq_node_units,
                           gsl::not_null<const NodeUnit*> q_node_unit) {
  assert(q_node_unit->OpType() == QDQ::QOpName);
  const size_t num_dqs = dq_node_units.size();
  if (num_dqs != 2 && num_dqs != 3) {
    return false;
  }

  // input and output types need to be same
  int32_t dt_input = dq_node_units[0]->GetNode().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_node_units[1]->GetNode().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_node_unit->GetNode().OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return false;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (dt_weight != dt_input) {
      return false;
    }
  }

  if (num_dqs == 3) {  // has bias
    int32_t dt_bias = dq_node_units[2]->GetNode().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
      return false;
    }
  }

  return true;
}

static Status QnnConvActivationFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                                         gsl::span<const NodeUnit*> dq_node_units,
                                         const NodeUnit* conv_node_unit,
                                         const NodeUnit* q_node_unit,
                                         const logging::Logger& logger,
                                         bool validate) {
  const size_t num_dqs = dq_node_units.size();
  constexpr size_t max_num_dqs = 3;
  ORT_RETURN_IF_NOT(num_dqs == 2 || num_dqs == max_num_dqs, "QDQ Conv should have 2 or 3 DQs");

  std::array<const Node*, max_num_dqs> dq_nodes_buf = {};
  for (size_t i = 0; i < num_dqs; i++) {
    dq_nodes_buf[i] = &dq_node_units[i]->GetNode();
  }
  gsl::span<const Node*> dq_nodes(dq_nodes_buf.data(), num_dqs);

  std::array<const Node*, 1> q_nodes = {&q_node_unit->GetNode()};
  const Node& target_node = conv_node_unit->GetNode();

  // Populate NodeUnit inputs
  std::vector<NodeUnitIODef> inputs;
  inputs.reserve(num_dqs);
  for (const Node* dq_node : dq_nodes) {
    const auto dq_inputs = dq_node->InputDefs();
    const auto& dq_attrs = dq_node->GetAttributes();

    std::optional<int64_t> axis;
    if (auto entry = dq_attrs.find("axis"); entry != dq_attrs.end()) {
      axis = entry->second.i();
    }

    // quantization scale and zp are always the input[1, 2]
    NodeUnitIODef::QuantParam quant_param{*dq_inputs[1], dq_inputs.size() == 3 ? dq_inputs[2] : nullptr, axis};
    inputs.push_back(NodeUnitIODef{*dq_inputs[0], quant_param});
  }

  // Populate NodeUnit outputs and output edges
  std::vector<NodeUnitIODef> outputs;
  Node::EdgeSet output_edges;
  for (const Node* q_node : q_nodes) {
    const auto q_inputs = q_node->InputDefs();
    const auto& q_attrs = q_node->GetAttributes();
    const auto q_outputs = q_node->OutputDefs();

    std::optional<int64_t> axis;
    if (auto entry = q_attrs.find("axis"); entry != q_attrs.end()) {
      axis = entry->second.i();
    }

    // quantization scale and zp are always the input[1, 2]
    NodeUnitIODef::QuantParam quant_param{*q_inputs[1], q_inputs.size() == 3 ? q_inputs[2] : nullptr, axis};
    outputs.push_back(NodeUnitIODef{*q_outputs[0], quant_param});

    // Gather output edges out of the Q node.
    auto q_cur_edge = q_node->OutputEdgesBegin();
    auto q_end_edge = q_node->OutputEdgesEnd();
    for (; q_cur_edge != q_end_edge; ++q_cur_edge) {
      output_edges.insert(Node::EdgeEnd{q_cur_edge->GetNode(), 0, q_cur_edge->GetDstArgIndex()});
    }
  }

  NodeUnit custom_node_unit(dq_nodes, target_node, q_nodes, NodeUnit::Type::QDQGroup,
                            inputs, outputs, num_dqs, output_edges);
  const auto* conv_op_builder = qnn::GetOpBuilder(custom_node_unit.OpType());
  if (conv_op_builder == nullptr) {
    return Status::OK();
  }

  if (validate) {
    return conv_op_builder->IsOpSupported(qnn_model_wrapper, custom_node_unit, logger);
  }

  return conv_op_builder->AddToModelBuilder(qnn_model_wrapper, custom_node_unit, logger, validate);
}

std::optional<QnnNodeGroup> TryConvActivationFusion(QnnModelWrapper& qnn_model_wrapper,
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
    return std::nullopt;
  }

  // Relu/Clip must have a single Q child.
  const std::array<std::string_view, 1> q_op_types = {QDQ::QOpName};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, *activation_node_unit, q_op_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    return std::nullopt;
  }

  // Check if Clip/Relu can be removed because the Q node provides an equivalent effect.
  if (!CanActivationBeRemoved(qnn_model_wrapper, *activation_node_unit, *q_node_unit)) {
    return std::nullopt;
  }

  // Create a QDQ node group with DQ* -> Conv -> Q
  const Node& conv_node = conv_node_unit.GetNode();
  const Node& activation_node = activation_node_unit->GetNode();
  std::vector<const NodeUnit*> dq_node_units = GetConvDQs(graph_viewer,
                                                          node_to_node_unit,
                                                          node_unit_to_qnn_node_group,
                                                          conv_node);

  if (!IsValidQDQConv(dq_node_units, q_node_unit)) {
    return std::nullopt;
  }

  LOGS(logger, VERBOSE) << "Will use Conv + Activation via fusion. conv_node name: [" << conv_node.Name()
                        << "] activation_node optype: [" << activation_node.OpType()
                        << "] activation_node name: [" << activation_node.Name()
                        << "]";

  std::optional<QnnNodeGroup> qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::ConvActivationFusion;
  qnn_node_group->node_units_ = std::move(dq_node_units);
  qnn_node_group->node_units_.push_back(&conv_node_unit);
  qnn_node_group->node_units_.push_back(activation_node_unit);
  qnn_node_group->node_units_.push_back(q_node_unit);

  return qnn_node_group;
}

namespace conv_act_fusion {

Status IsSupported(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  const size_t num_node_units = qnn_node_group.node_units_.size();
  ORT_RETURN_IF_NOT((num_node_units == 5 || num_node_units == 6), "");

  const bool has_bias_dq = num_node_units == 6;
  std::vector<const NodeUnit*> dq_node_units = {qnn_node_group.node_units_[0], qnn_node_group.node_units_[1]};
  const NodeUnit* conv_node_unit = qnn_node_group.node_units_[num_node_units - 3];
  const NodeUnit* activation_node_unit = qnn_node_group.node_units_[num_node_units - 2];
  const NodeUnit* q_node_unit = qnn_node_group.node_units_[num_node_units - 1];

  if (has_bias_dq) {
    dq_node_units.push_back(qnn_node_group.node_units_[2]);
  }
  Status status = QnnConvActivationFusionAdd(qmw,
                                             dq_node_units,
                                             conv_node_unit,
                                             q_node_unit,
                                             logger,
                                             /*validate*/ true);

  if (!status.IsOK()) {
    LOGS(logger, ERROR) << conv_node_unit->OpType() << "/" << activation_node_unit->OpType()
                        << " fusion is not supported, but should be according to initial validation."
                        << " Node names: " << conv_node_unit->Name() << ", " << activation_node_unit->Name()
                        << " Error: " << status.ErrorMessage();
  }

  return status;
}

Status AddToModelBuilder(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  const size_t num_node_units = qnn_node_group.node_units_.size();
  ORT_RETURN_IF_NOT((num_node_units == 5 || num_node_units == 6), "");

  const bool has_bias_dq = num_node_units == 6;
  std::vector<const NodeUnit*> dq_node_units = {qnn_node_group.node_units_[0], qnn_node_group.node_units_[1]};
  const NodeUnit* conv_node_unit = qnn_node_group.node_units_[num_node_units - 3];
  const NodeUnit* q_node_unit = qnn_node_group.node_units_[num_node_units - 1];

  if (has_bias_dq) {
    dq_node_units.push_back(qnn_node_group.node_units_[2]);
  }
  return QnnConvActivationFusionAdd(qmw,
                                    dq_node_units,
                                    conv_node_unit,
                                    q_node_unit,
                                    logger,
                                    /*validate*/ false);
}

#if 0
const std::vector<const NodeUnit*>& GetNodeUnits(const QnnNodeGroup& qnn_node_group) {
  return qnn_node_group.node_units_;
}
#endif

const NodeUnit* GetTargetNodeUnit(const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  const size_t num_node_units = qnn_node_group.node_units_.size();
  if (!(num_node_units == 5 || num_node_units == 6)) {
    return nullptr;
  }
  return qnn_node_group.node_units_[num_node_units - 3];
}
}  // namespace conv_act_fusion
}  // namespace qnn
}  // namespace onnxruntime
