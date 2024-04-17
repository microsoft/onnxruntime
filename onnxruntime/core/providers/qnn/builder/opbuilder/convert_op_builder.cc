// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "QnnOpDef.h"  // From QNN SDK: contains QNN constants (e.g., op names, param values).

namespace onnxruntime {
namespace qnn {

class ConvertOpBuilder : public BaseOpBuilder {
 public:
  ConvertOpBuilder() : BaseOpBuilder("ConvertOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvertOpBuilder);

  Status AddConvertToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& dq_node_unit,
                                  const NodeUnit& q_node_unit,
                                  const logging::Logger& logger,
                                  bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Status ConvertOpBuilder::AddConvertToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& dq_node_unit,
                                                  const NodeUnit& q_node_unit,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  std::vector<std::string> input_names;

  // Process the input from the DQ node
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, dq_node_unit.Inputs()[0], logger, input_names));

  // Process the output from the Q node. Override the QNN operator type to "Convert".
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, q_node_unit, std::move(input_names), {},
                                     logger, do_op_validation, QNN_OP_CONVERT));
  return Status::OK();
}

HandleConvertResult TryHandleConvertSequence(QnnModelWrapper& qnn_model_wrapper,
                                             const NodeUnit& maybe_dq_node_unit,
                                             const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                             const logging::Logger& logger,
                                             bool do_op_validation) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Looking for a standalone DQ to start the sequence.
  if (maybe_dq_node_unit.OpType() != QDQ::DQOpName || maybe_dq_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return {};
  }

  const Node& dq_node = maybe_dq_node_unit.GetNode();

  // DQ must have a single Q child. DQ must not produce a graph output.
  auto children = graph_utils::FindChildrenByType(dq_node, QDQ::QOpName);
  if (children.size() != 1 || dq_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(dq_node)) {
    return {};
  }

  const Node& q_node = *children[0];
  const auto q_node_unit_it = node_unit_map.find(&q_node);

  if (q_node_unit_it == node_unit_map.end()) {
    return {ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node does not have a corresponding NodeUnit"), nullptr};
  }

  const NodeUnit* q_node_unit = q_node_unit_it->second;

  // Q child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (q_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return {};
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  // DQ and Q must have equal scale type and different zp type.
  if (!QDQ::IsDQQConversion(dq_node, q_node, get_const_initializer, graph_viewer.ModelPath())) {
    return {};
  }

  ConvertOpBuilder op_builder;

  LOGS(logger, VERBOSE) << " Adding QNN Convert. dq_node name: [" << dq_node.Name()
                        << "] dq_node optype: [" << dq_node.OpType()
                        << "] q_node name: [" << q_node_unit->Name()
                        << "] q_node optype: [" << q_node_unit->OpType()
                        << "]";

  auto status = op_builder.AddConvertToModelBuilder(qnn_model_wrapper, maybe_dq_node_unit, *q_node_unit, logger,
                                                    do_op_validation);
  return status.IsOK() ? HandleConvertResult{status, q_node_unit} : HandleConvertResult{status, nullptr};
}

}  // namespace qnn
}  // namespace onnxruntime
