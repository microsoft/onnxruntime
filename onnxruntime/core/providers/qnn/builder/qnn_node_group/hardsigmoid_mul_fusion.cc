#include "core/providers/qnn/builder/qnn_node_group/hardsigmoid_mul_fusion.h"

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

std::optional<QnnNodeGroup> TryHardSigmoidMulFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& hardsigmoid_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // Looking for a standalone HardSigmoid to start the sequence.
  if (hardsigmoid_node_unit.OpType() != "HardSigmoid" ||
      hardsigmoid_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return std::nullopt;
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
    return std::nullopt;
  }

  // HardSigmoid must have a single Mul child (1 output edge) and must not produce a graph output.
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::array<std::string_view, 1> child_types = {"Mul"};
  const NodeUnit* mul_node_unit = GetOnlyChildOfType(graph_viewer, hardsigmoid_node_unit, child_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);

  if (mul_node_unit == nullptr) {
    return std::nullopt;
  }

  // Input to HardSigmoid must also be the other input to the Mul.
  const Node& mul_node = mul_node_unit->GetNode();
  auto& hs_input_name = hardsigmoid_node_unit.Inputs()[0].node_arg.Name();
  const bool same_root_input = mul_node.InputDefs()[0]->Name() == hs_input_name ||
                               mul_node.InputDefs()[1]->Name() == hs_input_name;

  if (!same_root_input) {
    return std::nullopt;
  }

  if (Status status = QnnHardSigmoidMulFusionAdd(qnn_model_wrapper, hardsigmoid_node_unit, *mul_node_unit,
                                                 logger, /*validate*/ true);
      !status.IsOK()) {
    return std::nullopt;
  }

  // Validation passed, so create a QnnNodeGroup. Any errors are now passed back to the caller.
  LOGS(logger, VERBOSE) << "Will use QNN HardSwish via fusion. HardSigmoid name: [" << hardsigmoid_node_unit.Name()
                        << "] Mul name: [" << mul_node_unit->Name() << "]";

  std::optional<QnnNodeGroup> qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::HardSigmoidMulFusion;
  qnn_node_group->node_units_.push_back(&hardsigmoid_node_unit);
  qnn_node_group->node_units_.push_back(mul_node_unit);

  return qnn_node_group;
}

namespace hs_mul_fusion {

Status IsSupported(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 2, "Expected 2 NodeUnits for HardSimoid -> Mul fusion");
  const NodeUnit* hardsigmoid_node_unit = qnn_node_group.node_units_[0];
  const NodeUnit* mul_node_unit = qnn_node_group.node_units_[1];
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

Status AddToModelBuilder(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 2, "Expected 2 NodeUnits for HardSimoid -> Mul fusion");
  const NodeUnit* hardsigmoid_node_unit = qnn_node_group.node_units_[0];
  const NodeUnit* mul_node_unit = qnn_node_group.node_units_[1];
  ORT_RETURN_IF_NOT(hardsigmoid_node_unit != nullptr && mul_node_unit != nullptr, "");
  return QnnHardSigmoidMulFusionAdd(qmw, *hardsigmoid_node_unit, *mul_node_unit, logger, /*validate*/ false);
}

#if 0
const std::vector<const NodeUnit*>& GetNodeUnits(const QnnNodeGroup& qnn_node_group) {
  return qnn_node_group.node_units_;
}
#endif

const NodeUnit* GetTargetNodeUnit(const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  if (qnn_node_group.node_units_.size() != 2) {
    return nullptr;
  }
  return qnn_node_group.node_units_[0];
}

}  // namespace hs_mul_fusion
}  // namespace qnn
}  // namespace onnxruntime
