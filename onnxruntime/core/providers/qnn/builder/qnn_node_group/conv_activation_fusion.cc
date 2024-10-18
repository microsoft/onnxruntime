#include "core/providers/qnn/builder/qnn_node_group/conv_activation_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include "core/graph/graph_utils.h"
#include "core/framework/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

namespace {

// 1. Clip can be removed iff the codoamin of QuantizeLinear remain unchanged.
// 2. To remain unchanged, y=QuantizeLinear(Clip(x)) must span the full range of values that can be represented by the
//    integer type of y. We can use this precondition to eval QuantizeLinear backward to to get the domain.
// 3. Indicates the domain of QuantizeLinear is strict subset of the codomain of Clip. We can use this to test if a
//    removal is valid or not.
// 4. Due to rounding effect, we can be get the upperbound and lowerbound of min or max
//    - Which one to use?
//      upperbound of min and lowerbound of max
//    - Why?
//      We want the codomain to be unchanged, so as long as the domain genreate a codomain that fill the integer value
//      range will be fine.

// The quantization formula is y = saturate((x / y_scale) + y_zero_point)
// For (x / y_scale), it rounds to the nearest even. So the allowed quantize limits before saturate need to be taken
// care of.
//
// The following struct provides a wrapper to compute the domain from codomain (Q output dtype).
template <typename T>
struct quantize_domain {
  static float min_upper(float scale, int zp) {
    int64_t codomain_min = std::numeric_limits<T>::lowest();
    int64_t biased = codomain_min - zp;
    float before_round = float(biased) + 0.5f;  // move to upperbound
    if (biased % 2 == 1) {                      // cannot be exact ?.5 because of rounding to even
      before_round = std::nextafterf(before_round, float(biased));
    }
    return before_round * scale;
  }

  static float max_lower(float scale, int zp) {
    int64_t codomain_max = std::numeric_limits<T>::max();
    int64_t biased = codomain_max - zp;
    float before_round = float(biased) - 0.5f;  // move to lowerbound
    if (biased % 2 == 1) {                      // cannot be exact ?.5 because of rounding to even
      before_round = std::nextafterf(before_round, float(biased));
    }
    return before_round * scale;
  }
};

}  // namespace

// Gets the scale, zero-point, and zero-point type for a QuantizeLinear node that uses per-tensor quantization.
static bool GetQScalarScaleZeroPoint(const QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& q_node_unit,
                                     /*out*/ float& scale,
                                     /*out*/ int32_t& zero_point,
                                     /*out*/ int32_t& zp_data_type) {
  assert(q_node_unit.OpType() == QUANTIZE_LINEAR);
  const auto& q_inputs = q_node_unit.GetNode().InputDefs();

  // Require an explicit zero-point input for now.
  if (q_inputs.size() != 3 || !q_inputs[QDQ_ZERO_POINT_INPUT_IDX]->Exists()) {
    return false;
  }

  std::vector<int32_t> zero_points;
  Status status = qnn_model_wrapper.UnpackZeroPoints(q_inputs[QDQ_ZERO_POINT_INPUT_IDX]->Name(),
                                                     zero_points, zp_data_type);

  // Should only have one zero-point (per-tensor).
  if (!status.IsOK() || zero_points.size() != 1) {
    return false;
  }
  zero_point = -zero_points[0];  // QNN zero-points are negated.

  std::vector<float> scales;
  status = qnn_model_wrapper.UnpackScales(q_inputs[QDQ_SCALE_INPUT_IDX]->Name(), scales);

  // Should only have one scale (per-tensor).
  if (!status.IsOK() || scales.size() != 1) {
    return false;
  }

  scale = scales[0];
  return true;
}

// Computes the floating point domain (min_, max_) from a QuantizeLinear node's scale/zero-point.
static bool GetQDomain(const QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& q_node_unit,
                       /*out*/ float& min_,
                       /*out*/ float& max_) {
  int32_t zp_data_type = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
  int32_t zero_point = 0;
  float scale = 0.0f;

  if (!GetQScalarScaleZeroPoint(qnn_model_wrapper, q_node_unit, scale, zero_point, zp_data_type)) {
    return false;
  }

  switch (zp_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      min_ = quantize_domain<int8_t>::min_upper(scale, zero_point);
      max_ = quantize_domain<int8_t>::max_lower(scale, zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      min_ = quantize_domain<uint8_t>::min_upper(scale, zero_point);
      max_ = quantize_domain<uint8_t>::max_lower(scale, zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      min_ = quantize_domain<int16_t>::min_upper(scale, zero_point);
      max_ = quantize_domain<int16_t>::max_lower(scale, zero_point);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      min_ = quantize_domain<uint16_t>::min_upper(scale, zero_point);
      max_ = quantize_domain<uint16_t>::max_lower(scale, zero_point);
      break;
    }
    default:
      return false;
  }

  return true;
}

// Returns true if the Clip in the sequence (Clip -> Q) can be removed because it is made redundant by the Q.
static bool CanClipBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& clip_node_unit,
                             const NodeUnit& q_node_unit,
                             const logging::Logger& logger) {
  assert(clip_node_unit.OpType() == "Clip" && q_node_unit.OpType() == QUANTIZE_LINEAR);
  float min_ = 0.0f;
  float max_ = 0.0f;

  if (!GetQDomain(qnn_model_wrapper, q_node_unit, min_, max_)) {
    return false;
  }

  float clip_min = std::numeric_limits<float>::lowest();
  float clip_max = std::numeric_limits<float>::max();

  if (!onnxruntime::GetClipMinMax(qnn_model_wrapper.GetGraphViewer(), clip_node_unit.GetNode(),
                                  clip_min, clip_max, logger)) {
    return false;
  }

  // The Clip codomain must entirely overlap the QuantizeLinear domain (quantization can be smaller).
  // Clip codomain:          [------------------]
  // QuantizeLinear domain:    [-------------]
  if (clip_min <= min_ && max_ <= clip_max) {
    return true;
  }

  return false;
}

// Returns true if the Relu in the sequence (Relu -> Q) can be removed because it is made redundant by the Q.
static bool CanQRelaceRelu(const QnnModelWrapper& qnn_model_wrapper, const NodeUnit& q_node_unit) {
  assert(q_node_unit.OpType() == QUANTIZE_LINEAR);
  int32_t zp_data_type = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;
  int32_t zero_point = 0;
  float scale = 0.0f;

  if (!GetQScalarScaleZeroPoint(qnn_model_wrapper, q_node_unit, scale, zero_point, zp_data_type)) {
    return false;
  }

  switch (zp_data_type) {
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT8:
      return quantize_domain<int8_t>::min_upper(scale, zero_point) >= 0;
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT8:
      return quantize_domain<uint8_t>::min_upper(scale, zero_point) >= 0;
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_INT16:
      return quantize_domain<int16_t>::min_upper(scale, zero_point) >= 0;
    case ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UINT16:
      return quantize_domain<uint16_t>::min_upper(scale, zero_point) >= 0;
    default:
      return false;
  }
}

// Returns true if the Clip/Relu in the sequence (Clip/Relu -> Q) can be removed because it is made redundant by the Q.
static bool CanActivationBeRemoved(const QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& activation_node_unit,
                                   const NodeUnit& q_node_unit,
                                   const logging::Logger& logger) {
  const std::string& activation_type = activation_node_unit.OpType();

  if (activation_type == "Relu") {
    return CanQRelaceRelu(qnn_model_wrapper, q_node_unit);
  }

  if (activation_type == "Clip") {
    return CanClipBeRemoved(qnn_model_wrapper, activation_node_unit, q_node_unit, logger);
  }

  return false;
}

// Returns the parent DQ nodes for a given node.
static std::vector<const Node*> FindParentDQNodes(const GraphViewer& graph_viewer, const Node& node) {
  // Get all parent DQ nodes sorted by destination argument index.
  std::vector<const Node*> parents(node.InputDefs().size(), nullptr);
  for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); it++) {
    if (it->GetNode().OpType().compare(DEQUANTIZE_LINEAR) == 0) {
      parents[it->GetDstArgIndex()] = &(it->GetNode());
    }
  }

  // Remove all the nodes which are not in the graph_viewer
  parents.erase(std::remove_if(parents.begin(), parents.end(),
                               [&graph_viewer](const Node* _node) {
                                 return _node == nullptr || graph_viewer.GetNode(_node->Index()) == nullptr;
                               }),
                parents.end());

  return parents;
}

// Gets the parent DQ nodes for the given Conv node. This fuction checks that the DQs are not a part of
// any other NodeUnit and that every Conv input comes from a parent DQ.
static bool GetConvDQs(
    const GraphViewer& graph_viewer,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Node& conv_node,
    /*out*/ std::array<const NodeUnit*, 3>& dq_node_units) {
  if (conv_node.OpType() != "Conv" && conv_node.OpType() != "ConvTranspose") {
    return false;
  }

  // Count number of inputs to Conv node.
  const auto& conv_inputs = conv_node.InputDefs();
  const size_t num_conv_inputs = std::count_if(conv_inputs.cbegin(), conv_inputs.cend(),
                                               [](const NodeArg* input) { return input && input->Exists(); });

  // Get the Conv's parent DQ nodes.
  std::vector<const Node*> dq_nodes = FindParentDQNodes(graph_viewer, conv_node);
  const size_t num_dqs = dq_nodes.size();

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  if ((num_conv_inputs != num_dqs) || (num_dqs > dq_node_units.size())) {
    return false;
  }

  dq_node_units.fill(nullptr);
  for (size_t i = 0; i < num_dqs; i++) {
    const Node* dq_node = dq_nodes[i];

    // DQ must not produce a graph output.
    if (!dq_node || graph_viewer.NodeProducesGraphOutput(*dq_node)) {
      return false;
    }

    // Conv should be the only consumer of a parent DQ.
    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == conv_node.Index();
    if (!dq_has_single_output_edge_to_target) {
      return false;
    }

    // DQ node must be part of a "standalone" NodeUnit.
    const auto it = node_to_node_unit.find(dq_node);
    if (it == node_to_node_unit.end()) {
      return false;
    }
    const NodeUnit* dq_node_unit = it->second;
    if (!dq_node_unit || node_unit_to_qnn_node_group.count(dq_node_unit) != 0) {
      return false;
    }
    if (dq_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return false;
    }

    dq_node_units[i] = dq_node_unit;
  }

  return true;
}

// Checks that the input and output data types are valid for a QDQ Conv.
static bool CheckQDQConvDataTypes(std::array<const NodeUnit*, 3>& dq_node_units,
                                  gsl::not_null<const NodeUnit*> q_node_unit) {
  assert(q_node_unit->OpType() == QUANTIZE_LINEAR);
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

  if (dq_node_units[2] != nullptr) {  // has bias
    int32_t dt_bias = dq_node_units[2]->GetNode().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
      return false;
    }
  }

  return true;
}

// Utility function to either validate or create a quantized QNN Conv node. The function creates a temporary
// custom NodeUnit that excludes the Clip/Relu because it is redundant. This custom NodeUnit is passed to our
// existing Conv OpBuilder for creation or validation via QNN APIs.
#define ValidateOnQnn(qnn_model_wrapper, dq_node_units, conv_node_unit, q_node_unit, logger) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_units), (conv_node_unit), (q_node_unit), (logger), true)
#define CreateOnQnn(qnn_model_wrapper, dq_node_units, conv_node_unit, q_node_unit, logger) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_units), (conv_node_unit), (q_node_unit), (logger), false)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> dq_node_units,
                                    const NodeUnit* conv_node_unit,
                                    const NodeUnit* q_node_unit,
                                    const logging::Logger& logger,
                                    bool validate) {
  const size_t num_dqs = dq_node_units.size();
  constexpr size_t max_num_dqs = 3;
  ORT_RETURN_IF_NOT(num_dqs == 2 || num_dqs == max_num_dqs, "QDQ Conv should have 2 or 3 DQs");
  ORT_RETURN_IF_NOT(conv_node_unit->OpType() == "Conv" && q_node_unit->OpType() == QUANTIZE_LINEAR,
                    "Expected Conv/ConvTranspose and QuantizeLinear but got ", conv_node_unit->OpType(), " and ",
                    q_node_unit->OpType());

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

// Traverses graph to check if the given NodeUnit is part of a valid DQ* -> Conv -> Relu/Clip -> Q sequence.
// If so, returns a IQnnNodeGroup that contains the constituent NodeUnits.
std::unique_ptr<IQnnNodeGroup> ConvActivationFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& conv_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // Expect that this function is called with a standalone Conv or ConvTranspose.
  const auto& conv_type = conv_node_unit.OpType();

  if ((conv_type != "Conv" && conv_type != "ConvTranspose") ||
      (conv_node_unit.UnitType() != NodeUnit::Type::SingleNode)) {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Conv must have a single Relu or Clip child.
  const std::array<std::string_view, 2> activation_op_types = {"Relu", "Clip"};
  const NodeUnit* activation_node_unit = GetOnlyChildOfType(graph_viewer, conv_node_unit, activation_op_types,
                                                            node_to_node_unit, node_unit_to_qnn_node_group);
  if (activation_node_unit == nullptr) {
    return nullptr;
  }

  // Relu/Clip must have a single Q child.
  const std::array<std::string_view, 1> q_op_types = {QUANTIZE_LINEAR};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, *activation_node_unit, q_op_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    return nullptr;
  }

  // Check if Clip/Relu can be removed because the Q node provides an equivalent effect.
  if (!CanActivationBeRemoved(qnn_model_wrapper, *activation_node_unit, *q_node_unit, logger)) {
    return nullptr;
  }

  // Create a QDQ node group with DQ* -> Conv -> Q
  const Node& conv_node = conv_node_unit.GetNode();
  std::array<const NodeUnit*, 3> dq_node_units = {};
  if (!GetConvDQs(graph_viewer,
                  node_to_node_unit,
                  node_unit_to_qnn_node_group,
                  conv_node, dq_node_units)) {
    return nullptr;
  }

  if (!CheckQDQConvDataTypes(dq_node_units, q_node_unit)) {
    return nullptr;
  }

  return std::make_unique<ConvActivationFusion>(*dq_node_units[0],
                                                *dq_node_units[1],
                                                dq_node_units[2],
                                                conv_node_unit,
                                                *activation_node_unit,
                                                *q_node_unit);
}

ConvActivationFusion::ConvActivationFusion(const NodeUnit& dq_node_unit_0,
                                           const NodeUnit& dq_node_unit_1,
                                           const NodeUnit* dq_node_unit_2,
                                           const NodeUnit& conv_node_unit,
                                           const NodeUnit& activation_node_unit,
                                           const NodeUnit& q_node_unit)
    : node_units_{} {
  size_t i = 0;
  node_units_[i++] = &dq_node_unit_0;
  node_units_[i++] = &dq_node_unit_1;
  if (dq_node_unit_2 != nullptr) {
    node_units_[i++] = dq_node_unit_2;
  }
  node_units_[i++] = &conv_node_unit;
  node_units_[i++] = &activation_node_unit;
  node_units_[i++] = &q_node_unit;
  assert((!dq_node_unit_2 && i == 5) || (dq_node_unit_2 && i == 6));
}

Status ConvActivationFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  const size_t num_dqs = node_units_.back() != nullptr ? 3 : 2;
  gsl::span<const NodeUnit* const> dq_node_units(node_units_.data(), num_dqs);

  return ValidateOnQnn(qmw, dq_node_units,
                       node_units_[num_dqs],      // Conv
                       node_units_[num_dqs + 2],  // Q
                       logger);
}

Status ConvActivationFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  const size_t num_dqs = node_units_.back() != nullptr ? 3 : 2;
  gsl::span<const NodeUnit* const> dq_node_units(node_units_.data(), num_dqs);

  return CreateOnQnn(qmw, dq_node_units,
                     node_units_[num_dqs],      // Conv
                     node_units_[num_dqs + 2],  // Q
                     logger);
}

gsl::span<const NodeUnit* const> ConvActivationFusion::GetNodeUnits() const {
  const size_t num_node_units = node_units_.back() != nullptr ? 6 : 5;
  return gsl::make_span<const NodeUnit* const>(node_units_.data(), num_node_units);
}

const NodeUnit* ConvActivationFusion::GetTargetNodeUnit() const {
  const size_t conv_index = node_units_.back() != nullptr ? 3 : 2;
  return node_units_[conv_index];
}

}  // namespace qnn
}  // namespace onnxruntime
