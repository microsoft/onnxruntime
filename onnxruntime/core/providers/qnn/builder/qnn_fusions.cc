// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_fusions.h"

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_conv_activation_fusion.h"

namespace onnxruntime {
namespace qnn {

static Status QnnDQQFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                              const NodeUnit& dq_node_unit,
                              const NodeUnit& q_node_unit,
                              const logging::Logger& logger,
                              bool validate = false) {
  ORT_UNUSED_PARAMETER(logger);
  assert(dq_node_unit.OpType() == QDQ::DQOpName && q_node_unit.OpType() == QDQ::QOpName);
  const auto& node_name = utils::GetNodeName(dq_node_unit);
  const NodeUnitIODef& input_def = dq_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = q_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CONVERT,
                                                          {input_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(q_node_unit),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CONVERT,
                                                      {input_def.node_arg.Name()},
                                                      {output_def.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to add fused Convert node.");
  }

  return Status::OK();
}

static Status QnnHardSigmoidMulFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& hardsigmoid_node_unit,
                                         const NodeUnit& mul_node_unit,
                                         const logging::Logger& logger,
                                         bool validate = false) {
  ORT_UNUSED_PARAMETER(logger);
  assert(hardsigmoid_node_unit.OpType() == "HardSigmoid" && mul_node_unit.OpType() == "Mul");
  const auto& node_name = utils::GetNodeName(hardsigmoid_node_unit);
  const NodeUnitIODef& input_def = hardsigmoid_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = mul_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_HARD_SWISH,
                                                          {input_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    LOGS(logger, VERBOSE) << " Adding QNN HardSwish via fusion. HardSigmoid name: [" << hardsigmoid_node_unit.Name()
                          << "] Mul name: [" << mul_node_unit.Name() << "]";

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_HARD_SWISH,
                                                      {input_def.node_arg.Name()},
                                                      {output_def.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to add fused HardSwish node.");
  }

  return Status::OK();
}

std::string_view QnnNodeGroup::TypeToString(QnnNodeGroup::Type type) {
  static std::array<std::string_view, static_cast<size_t>(QnnNodeGroup::Type::COUNT)> type_names = {
      "Undefined",
      "NodeUnit",
      "ConvActivationFusion",
      "DQQFusion",
      "HardSigmoidMulFusion",
  };

  return type_names[static_cast<size_t>(type)];
}

Status QnnNodeGroup::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  switch (type_) {
    case Type::NodeUnit: {
      ORT_RETURN_IF_NOT(node_units_.size() == 1 && node_units_[0] != nullptr, "");
      const NodeUnit& node_unit = *node_units_[0];
      const std::string& op_type = node_unit.OpType();
      const auto* op_builder = qnn::GetOpBuilder(op_type);

      if (op_builder == nullptr) {
        std::string err_msg = MakeString("Operators of type `", op_type,
                                         "` are not supported by QNN EP.", op_type, " node `",
                                         node_unit.Name(), "` will not be assigned to QNN EP.");
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, err_msg);
      }

      Status status = op_builder->IsOpSupported(qmw, *node_units_[0], logger);
      if (!status.IsOK()) {
        LOGS(logger, WARNING) << op_type << " node `" << node_unit.Name()
                              << "` is not supported: " << status.ErrorMessage();
      }

      return status;
    }
    case Type::ConvActivationFusion: {
      const size_t num_node_units = node_units_.size();
      ORT_RETURN_IF_NOT((num_node_units == 5 || num_node_units == 6), "");

      const bool has_bias_dq = num_node_units == 6;
      std::vector<const NodeUnit*> dq_node_units = {node_units_[0], node_units_[1]};
      const NodeUnit* conv_node_unit = node_units_[num_node_units - 3];
      const NodeUnit* activation_node_unit = node_units_[num_node_units - 2];
      const NodeUnit* q_node_unit = node_units_[num_node_units - 1];

      if (has_bias_dq) {
        dq_node_units.push_back(node_units_[2]);
      }
      Status status = QnnConvActivationFusionAdd(qmw,
                                                 dq_node_units,
                                                 conv_node_unit,
                                                 activation_node_unit,
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
    case Type::DQQFusion: {
      ORT_RETURN_IF_NOT(node_units_.size() == 2, "Expected 2 NodeUnits for DQ -> Q fusion");
      const NodeUnit* dq_node_unit = node_units_[0];
      const NodeUnit* q_node_unit = node_units_[1];
      ORT_RETURN_IF_NOT(dq_node_unit != nullptr && q_node_unit != nullptr, "");
      Status status = QnnDQQFusionAdd(qmw, *dq_node_unit, *q_node_unit, logger, /*validate*/ true);

      if (!status.IsOK()) {
        LOGS(logger, ERROR) << "(DQ -> Q) into QNN Convert fusion is not supported, "
                            << "but should be according to initial validation. "
                            << "Node names: " << dq_node_unit->Name() << ", " << q_node_unit->Name()
                            << " Error: " << status.ErrorMessage();
      }

      return status;
    }
    case Type::HardSigmoidMulFusion: {
      ORT_RETURN_IF_NOT(node_units_.size() == 2, "Expected 2 NodeUnits for HardSimoid -> Mul fusion");
      const NodeUnit* hardsigmoid_node_unit = node_units_[0];
      const NodeUnit* mul_node_unit = node_units_[1];
      ORT_RETURN_IF_NOT(hardsigmoid_node_unit != nullptr && mul_node_unit != nullptr, "");
      Status status = QnnHardSigmoidMulFusionAdd(qmw, *hardsigmoid_node_unit, *mul_node_unit, logger,
                                                 /*validate*/ true);

      if (!status.IsOK()) {
        LOGS(logger, ERROR) << "(HardSigmoid -> Mul) into QNN HardSwish fusion is not supported, "
                            << "but should be according to initial validation. "
                            << "Node names: " << hardsigmoid_node_unit->Name() << ", " << mul_node_unit->Name()
                            << " Error: " << status.ErrorMessage();
      }

      return status;
    }
    default:
      std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(type_),
                                         " in QnnNodeGroup::IsSupported()");
      LOGS(logger, ERROR) << error_msg;
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
  }
}

Status QnnNodeGroup::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  switch (type_) {
    case Type::NodeUnit: {
      ORT_RETURN_IF_NOT(node_units_.size() == 1 && node_units_[0] != nullptr, "");
      const auto* op_builder = qnn::GetOpBuilder(node_units_[0]->OpType());
      ORT_RETURN_IF_NOT(op_builder != nullptr, "[QNN EP]: Missing OpBuilder for OpType ", node_units_[0]->OpType());
      return op_builder->AddToModelBuilder(qmw, *node_units_[0], logger, /*do_op_validation*/ false);
    }
    case Type::ConvActivationFusion: {
      const size_t num_node_units = node_units_.size();
      ORT_RETURN_IF_NOT((num_node_units == 5 || num_node_units == 6), "");

      const bool has_bias_dq = num_node_units == 6;
      std::vector<const NodeUnit*> dq_node_units = {node_units_[0], node_units_[1]};
      const NodeUnit* conv_node_unit = node_units_[num_node_units - 3];
      const NodeUnit* activation_node_unit = node_units_[num_node_units - 2];
      const NodeUnit* q_node_unit = node_units_[num_node_units - 1];

      if (has_bias_dq) {
        dq_node_units.push_back(node_units_[2]);
      }
      return QnnConvActivationFusionAdd(qmw,
                                        dq_node_units,
                                        conv_node_unit,
                                        activation_node_unit,
                                        q_node_unit,
                                        logger,
                                        /*validate*/ false);
    }
    case Type::DQQFusion: {
      ORT_RETURN_IF_NOT(node_units_.size() == 2, "Expected 2 NodeUnits for DQ -> Q fusion");
      const NodeUnit* dq_node_unit = node_units_[0];
      const NodeUnit* q_node_unit = node_units_[1];
      ORT_RETURN_IF_NOT(dq_node_unit != nullptr && q_node_unit != nullptr, "");
      return QnnDQQFusionAdd(qmw, *dq_node_unit, *q_node_unit, logger, /*validate*/ false);
    }
    case Type::HardSigmoidMulFusion: {
      ORT_RETURN_IF_NOT(node_units_.size() == 2, "Expected 2 NodeUnits for HardSimoid -> Mul fusion");
      const NodeUnit* hardsigmoid_node_unit = node_units_[0];
      const NodeUnit* mul_node_unit = node_units_[1];
      ORT_RETURN_IF_NOT(hardsigmoid_node_unit != nullptr && mul_node_unit != nullptr, "");
      return QnnHardSigmoidMulFusionAdd(qmw, *hardsigmoid_node_unit, *mul_node_unit, logger, /*validate*/ false);
    }
    default:
      std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(type_),
                                         " in QnnNodeGroup::AddToModelBuilder()");
      LOGS(logger, ERROR) << error_msg;
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
  }
}

const NodeUnit* QnnNodeGroup::GetTargetNodeUnit(const logging::Logger& logger) const {
  switch (type_) {
    case Type::NodeUnit: {
      if (node_units_.size() != 1) {
        return nullptr;
      }
      return node_units_[0];
    }
    case Type::ConvActivationFusion: {
      const size_t num_node_units = node_units_.size();
      if (!(num_node_units == 5 || num_node_units == 6)) {
        return nullptr;
      }
      return node_units_[num_node_units - 3];
    }
    case Type::DQQFusion: {
      if (node_units_.size() != 2) {
        return nullptr;
      }
      return node_units_[0];
    }
    case Type::HardSigmoidMulFusion: {
      if (node_units_.size() != 2) {
        return nullptr;
      }
      return node_units_[0];
    }
    default:
      std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(type_),
                                         " in QnnNodeGroup::AddToModelBuilder()");
      LOGS(logger, ERROR) << error_msg;
      return nullptr;
  }
}

/**
 * Tries to merge a DQ -> Q sequence into a QNN Convert operator. The DQ -> Q must be converting from
 * one quantization type (e.g., uint8_t) to another (e.g., uint16_t).
 *
 * \param fused_nodes Output list of node units that were fused. Remains empty if fusion is not applied.
 * \param qnn_model_wrapper The QNN model that is being built.
 * \param dq_node_unit The DQ node unit.
 * \param q_node_unit The Q node unit.
 * \param logger The logger.
 * \param do_op_validation True if should call QNN operator validation APIs.
 * \return An onnxruntime::Status
 */
static Status TryDQQFusion(std::optional<QnnNodeGroup>& qnn_node_group,
                           QnnModelWrapper& qnn_model_wrapper,
                           const NodeUnit& dq_node_unit,
                           const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                           const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
                           const logging::Logger& logger) {
  // Expect that this function is called with a standalone DQ.
  assert(dq_node_unit.OpType() == QDQ::DQOpName && dq_node_unit.UnitType() == NodeUnit::Type::SingleNode);

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& dq_node = dq_node_unit.GetNode();

  // DQ must have a single child (1 output edge) and must not produce a graph output.
  if (dq_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(dq_node)) {
    return Status::OK();
  }

  const Node& q_node = dq_node.OutputEdgesBegin()->GetNode();
  if (q_node.OpType() != QDQ::QOpName) {
    return Status::OK();
  }

  const auto q_node_unit_it = node_to_node_unit.find(&q_node);
  ORT_RETURN_IF(q_node_unit_it == node_to_node_unit.end(), "Node does not have a corresponding NodeUnit");
  const NodeUnit* q_node_unit = q_node_unit_it->second;

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (q_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  // Check if child node has already been handled. Should not be the case if this
  // fusion function has been called in topological order, but check to be safe.
  if (node_unit_to_qnn_node_group.count(q_node_unit) != 0) {
    return Status::OK();
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  // DQ and Q must have equal scale type and different zp type.
  if (!QDQ::IsDQQConversion(dq_node, q_node, get_const_initializer, graph_viewer.ModelPath())) {
    return Status::OK();
  }

  QNN_RETURN_OK_IF_ERROR(QnnDQQFusionAdd(qnn_model_wrapper, dq_node_unit, *q_node_unit, logger, /*validate*/ true),
                         logger);

  // Validation passed, so create a QnnNodeGroup.
  LOGS(logger, VERBOSE) << " Will use QNN Convert via fusion. dq_node name: [" << dq_node.Name()
                        << "] dq_node optype: [" << dq_node.OpType()
                        << "] q_node name: [" << q_node_unit->Name()
                        << "] q_node optype: [" << q_node_unit->OpType()
                        << "]";

  qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::DQQFusion;
  qnn_node_group->node_units_.push_back(&dq_node_unit);
  qnn_node_group->node_units_.push_back(q_node_unit);

  return Status::OK();
}

/**
 * Tries to fuse the sequence `x * HardSigmoid<alpha=1/6, beta=0.5>(x)` into a single HardSwish(x) operator.
 * Should be called in a topologically ordered iteration of node units.
 *
 * \param fused_nodes Output list of node units that were fused. Remains empty if fusion was not applied.
 * \param qnn_model_wrapper The QNN model that is being built.
 * \param starting_node The node unit that could potentially start the sequence.
 * \param node_unit_map Maps a node to its node unit.
 * \param handled_node_units Set of node units that have already been processed. Fusion will not fuse nodes
 *                           in this set.
 * \param logger The logger.
 * \param do_op_validation True if should call QNN operator validation APIs.
 * \return A Status indicating a potential failure.
 */
static Status TryHardSigmoidMulFusion(std::optional<QnnNodeGroup>& qnn_node_group,
                                      QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& hardsigmoid_node_unit,
                                      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                                      const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
                                      const logging::Logger& logger) {
  // Looking for a standalone HardSigmoid to start the sequence.
  if (hardsigmoid_node_unit.OpType() != "HardSigmoid" ||
      hardsigmoid_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  NodeAttrHelper hs_attr_helper(hardsigmoid_node_unit);
  float alpha = hs_attr_helper.Get("alpha", 0.2f);
  float beta = hs_attr_helper.Get("beta", 0.5f);
  constexpr float req_alpha = 1.0f / 6.0f;
  constexpr float req_beta = 0.5f;
  constexpr float alpha_eps = std::numeric_limits<float>::epsilon() * req_alpha;
  constexpr float beta_eps = std::numeric_limits<float>::epsilon() * req_beta;

  // Check for explicit values of alpha and beta.
  if (std::abs(alpha - req_alpha) > alpha_eps || std::abs(beta - req_beta) > beta_eps) {
    return Status::OK();
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& hs_node = hardsigmoid_node_unit.GetNode();

  // HardSigmoid must have a single child (1 output edge) and must not produce a graph output.
  if (hs_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(hs_node)) {
    return Status::OK();
  }

  const Node& mul_node = hs_node.OutputEdgesBegin()->GetNode();
  if (mul_node.OpType() != "Mul") {
    return Status::OK();
  }

  const auto mul_node_unit_it = node_to_node_unit.find(&mul_node);
  ORT_RETURN_IF(mul_node_unit_it == node_to_node_unit.end(), "Mul Node does not have a corresponding NodeUnit");
  const NodeUnit* mul_node_unit = mul_node_unit_it->second;

  // Check if Mul node has already been handled. Should not be the case if this
  // fusion function has been called in topological order, but check to be safe.
  if (node_unit_to_qnn_node_group.count(mul_node_unit) != 0) {
    return Status::OK();
  }

  // Mul child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (mul_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  // Input to HardSigmoid must also be the other input to the Mul.
  auto& hs_input_name = hardsigmoid_node_unit.Inputs()[0].node_arg.Name();
  const bool same_root_input = mul_node.InputDefs()[0]->Name() == hs_input_name ||
                               mul_node.InputDefs()[1]->Name() == hs_input_name;

  if (!same_root_input) {
    return Status::OK();
  }

  QNN_RETURN_OK_IF_ERROR(QnnHardSigmoidMulFusionAdd(qnn_model_wrapper, hardsigmoid_node_unit, *mul_node_unit,
                                                    logger, /*validate*/ true),
                         logger);

  // Validation passed, so create a QnnNodeGroup. Any errors are now passed back to the caller.
  LOGS(logger, VERBOSE) << "Will use QNN HardSwish via fusion. HardSigmoid name: [" << hardsigmoid_node_unit.Name()
                        << "] Mul name: [" << mul_node_unit->Name() << "]";

  qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::HardSigmoidMulFusion;
  qnn_node_group->node_units_.push_back(&hardsigmoid_node_unit);
  qnn_node_group->node_units_.push_back(mul_node_unit);

  return Status::OK();
}

using FusionFunc = Status (*)(std::optional<QnnNodeGroup>&,
                              QnnModelWrapper&,
                              const NodeUnit&,
                              const std::unordered_map<const Node*, const NodeUnit*>&,
                              const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>&,
                              const logging::Logger&);

static Status TryQnnFusions(/*out*/ std::optional<QnnNodeGroup>& fused_node_group,
                            QnnModelWrapper& qnn_model_wrapper,
                            const NodeUnit& starting_node_unit,
                            const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                            const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
                            const logging::Logger& logger) {
  // Maps a starting operator type to the fusion function.
  static std::unordered_map<std::string, FusionFunc> fusions = {
      {"DequantizeLinear", TryDQQFusion},
      {"HardSigmoid", TryHardSigmoidMulFusion},
      {"Conv", TryConvActivationFusion},
      {"ConvTranspose", TryConvActivationFusion},
  };

  // For now, all fusions involve standalone node units (i.e., no wrapping DQ/Q nodes).
  if (starting_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  auto iter = fusions.find(starting_node_unit.OpType());
  if (iter != fusions.end()) {
    FusionFunc fusion_func = iter->second;
    ORT_RETURN_IF_ERROR(fusion_func(fused_node_group, qnn_model_wrapper, starting_node_unit, node_to_node_unit,
                                    node_unit_to_qnn_node_group, logger));
  }
  return Status::OK();
}

Status GetQnnNodeGroups(/*out*/ std::vector<QnnNodeGroup>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                        const size_t num_node_units,
                        const logging::Logger& logger) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::vector<NodeIndex> sorted_node_indices = graph_viewer.GetNodesInTopologicalOrder();

  std::vector<QnnNodeGroup::IndexType> sorted_qnn_node_group_indices;
  sorted_qnn_node_group_indices.reserve(num_node_units);

  std::vector<QnnNodeGroup> tmp_qnn_node_groups;
  tmp_qnn_node_groups.reserve(num_node_units);

  {
    std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType> node_unit_to_qnn_node_group;
    std::vector<gsl::not_null<const NodeUnit*>> sorted_node_units;
    sorted_node_units.reserve(num_node_units);

    // Create QnnNodeGroups for fusions first.
    for (NodeIndex node_index : sorted_node_indices) {
      gsl::not_null<const Node*> node = graph_viewer.GetNode(node_index);

      // Get the NodeUnit associated with the node.
      const auto node_unit_it = node_to_node_unit.find(node);
      ORT_RETURN_IF_NOT(node_unit_it != node_to_node_unit.end(), "Could not find NodeUnit for Node ", node->Name());
      gsl::not_null<const NodeUnit*> node_unit = node_unit_it->second;

      // Skip this node if it is not the NodeUnit's target node to ensure NodeUnits are visited in topological order.
      if (node != &node_unit->GetNode()) {
        continue;
      }

      sorted_node_units.push_back(node_unit);

      if (node_unit_to_qnn_node_group.count(node_unit) != 0) {
        continue;  // Already handled this node unit
      }

      std::optional<QnnNodeGroup> fused_node_group;
      ORT_RETURN_IF_ERROR(TryQnnFusions(fused_node_group, qnn_model_wrapper, *node_unit,
                                        node_to_node_unit, node_unit_to_qnn_node_group, logger));

      if (fused_node_group.has_value()) {
        const QnnNodeGroup::IndexType index = tmp_qnn_node_groups.size();
        fused_node_group->index_ = index;

        for (const NodeUnit* fused_node_unit : fused_node_group->GetNodeUnits()) {
          assert(fused_node_unit != nullptr);
          node_unit_to_qnn_node_group.insert({fused_node_unit, index});
        }

        tmp_qnn_node_groups.push_back(std::move(*fused_node_group));
      }
    }

    // Create QnnNodeGroups for the leftover NodeUnits.
    for (gsl::not_null<const NodeUnit*> node_unit : sorted_node_units) {
      const auto it = node_unit_to_qnn_node_group.find(node_unit);
      if (it != node_unit_to_qnn_node_group.end()) {
        // Already handled this NodeUnit.
        const QnnNodeGroup& qnn_node_group = tmp_qnn_node_groups[it->second];
        if (node_unit == qnn_node_group.GetTargetNodeUnit(logger)) {
          sorted_qnn_node_group_indices.push_back(qnn_node_group.index_);
        }
        continue;
      }

      const QnnNodeGroup::IndexType index = tmp_qnn_node_groups.size();
      QnnNodeGroup fused_node_group = {};
      fused_node_group.type_ = QnnNodeGroup::Type::NodeUnit;
      fused_node_group.index_ = index;
      fused_node_group.node_units_.resize(1);
      fused_node_group.node_units_[0] = node_unit;
      tmp_qnn_node_groups.push_back(std::move(fused_node_group));

      node_unit_to_qnn_node_group.insert({node_unit, index});
      sorted_qnn_node_group_indices.push_back(index);
    }

    assert(tmp_qnn_node_groups.size() == sorted_qnn_node_group_indices.size());
  }

  // Copy QnnNodeGroups to output in sorted (topological) order.
  qnn_node_groups.resize(0);
  qnn_node_groups.reserve(tmp_qnn_node_groups.size());
  for (auto index : sorted_qnn_node_group_indices) {
    assert(index < tmp_qnn_node_groups.size());
    QnnNodeGroup qnn_node_group = std::move(tmp_qnn_node_groups[index]);
    qnn_node_group.index_ = qnn_node_groups.size();
    qnn_node_groups.push_back(std::move(qnn_node_group));
  }

  assert(qnn_node_groups.size() == sorted_qnn_node_group_indices.size());

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
