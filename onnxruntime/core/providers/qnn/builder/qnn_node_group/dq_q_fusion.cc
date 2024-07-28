#include "core/providers/qnn/builder/qnn_node_group/dq_q_fusion.h"

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

std::optional<QnnNodeGroup> TryDQQFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& dq_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // Expect that this function is called with a standalone DQ.
  if (dq_node_unit.OpType() != "DequantizeLinear" || dq_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& dq_node = dq_node_unit.GetNode();

  // DQ must have a single Q child (1 output edge) and must not produce a graph output.
  const std::array<std::string_view, 1> child_types = {"QuantizeLinear"};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, dq_node_unit, child_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    return std::nullopt;
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  // DQ and Q must have equal scale type and different zp type.
  if (!QDQ::IsDQQConversion(dq_node, q_node_unit->GetNode(), get_const_initializer, graph_viewer.ModelPath())) {
    return std::nullopt;
  }

  if (Status status = QnnDQQFusionAdd(qnn_model_wrapper, dq_node_unit, *q_node_unit,
                                      logger, /*validate*/ true);
      !status.IsOK()) {
    return std::nullopt;
  }

  // Validation passed, so create a QnnNodeGroup.
  LOGS(logger, VERBOSE) << " Will use QNN Convert via fusion. dq_node name: [" << dq_node.Name()
                        << "] dq_node optype: [" << dq_node.OpType()
                        << "] q_node name: [" << q_node_unit->Name()
                        << "] q_node optype: [" << q_node_unit->OpType()
                        << "]";

  std::optional<QnnNodeGroup> qnn_node_group = QnnNodeGroup{};
  qnn_node_group->type_ = QnnNodeGroup::Type::DQQFusion;
  qnn_node_group->node_units_.push_back(&dq_node_unit);
  qnn_node_group->node_units_.push_back(q_node_unit);

  return qnn_node_group;
}

namespace dq_q_fusion {

Status IsSupported(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 2, "Expected 2 NodeUnits for DQ -> Q fusion");
  const NodeUnit* dq_node_unit = qnn_node_group.node_units_[0];
  const NodeUnit* q_node_unit = qnn_node_group.node_units_[1];
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

Status AddToModelBuilder(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) {
  ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 2, "Expected 2 NodeUnits for DQ -> Q fusion");
  const NodeUnit* dq_node_unit = qnn_node_group.node_units_[0];
  const NodeUnit* q_node_unit = qnn_node_group.node_units_[1];
  ORT_RETURN_IF_NOT(dq_node_unit != nullptr && q_node_unit != nullptr, "");
  return QnnDQQFusionAdd(qmw, *dq_node_unit, *q_node_unit, logger, /*validate*/ false);
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

}  // namespace dq_q_fusion
}  // namespace qnn
}  // namespace onnxruntime
