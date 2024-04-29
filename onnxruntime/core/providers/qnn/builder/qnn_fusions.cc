// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_fusions.h"

#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

static Status AddTensor(QnnModelWrapper& qnn_model_wrapper,
                        const NodeUnitIODef& tensor,
                        std::vector<std::string>& tensor_names) {
  const std::string& tensor_name = tensor.node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(tensor_name)) {
    tensor_names.push_back(tensor_name);
    return Status::OK();
  }

  TensorInfo tensor_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(tensor, tensor_info));

  std::vector<uint8_t> unpacked_tensor;
  if (tensor_info.is_initializer) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*tensor_info.initializer_tensor, unpacked_tensor));
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(tensor_name);
  QnnTensorWrapper tensor_wrapper(tensor_name, tensor_type, tensor_info.qnn_data_type,
                                  std::move(tensor_info.quant_param), std::move(tensor_info.shape),
                                  std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)), "Failed to add tensor.");
  tensor_names.push_back(tensor_name);

  return Status::OK();
}

/**
 * Tries to merge a DQ -> Q sequence into a QNN Convert operator. The DQ -> Q must be converting from
 * one quantization type (e.g., uint8_t) to another (e.g., uint16_t).
 *
 * \param fused_nodes Output list of node units that were fused. Remains empty if fusion is not applied.
 * \param qnn_model_wrapper The QNN model that is being built.
 * \param maybe_dq_node_unit The node unit that could potentially start the DQ -> Q sequence.
 * \param logger The logger.
 * \param do_op_validation True if should call QNN operator validation APIs.
 * \return An onnxruntime::Status
 */
static Status TryHandleConvertSequence(std::vector<const NodeUnit*>& fused_nodes,
                                       QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& maybe_dq_node_unit,
                                       const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                       const logging::Logger& logger,
                                       bool do_op_validation) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Looking for a standalone DQ to start the sequence.
  if (maybe_dq_node_unit.OpType() != QDQ::DQOpName || maybe_dq_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  const Node& dq_node = maybe_dq_node_unit.GetNode();

  // DQ must have a single Q child. DQ must not produce a graph output.
  auto children = graph_utils::FindChildrenByType(dq_node, QDQ::QOpName);
  if (children.size() != 1 || dq_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(dq_node)) {
    return Status::OK();
  }

  const Node& q_node = *children[0];
  const auto q_node_unit_it = node_unit_map.find(&q_node);

  ORT_RETURN_IF(q_node_unit_it == node_unit_map.end(), "Node does not have a corresponding NodeUnit");

  const NodeUnit* q_node_unit = q_node_unit_it->second;

  // Q child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (q_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  // DQ and Q must have equal scale type and different zp type.
  if (!QDQ::IsDQQConversion(dq_node, q_node, get_const_initializer, graph_viewer.ModelPath())) {
    return Status::OK();
  }

  LOGS(logger, VERBOSE) << " Adding QNN Convert. dq_node name: [" << dq_node.Name()
                        << "] dq_node optype: [" << dq_node.OpType()
                        << "] q_node name: [" << q_node_unit->Name()
                        << "] q_node optype: [" << q_node_unit->OpType()
                        << "]";

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Add the input from the DQ node, and the output from the Q node.
  ORT_RETURN_IF_ERROR(AddTensor(qnn_model_wrapper, maybe_dq_node_unit.Inputs()[0], input_names));
  ORT_RETURN_IF_ERROR(AddTensor(qnn_model_wrapper, q_node_unit->Outputs()[0], output_names));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(*q_node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CONVERT,
                                                    std::move(input_names),
                                                    std::move(output_names),
                                                    {},
                                                    do_op_validation),
                    "Failed to add Convert node.");

  fused_nodes.push_back(&maybe_dq_node_unit);
  fused_nodes.push_back(q_node_unit);

  return Status::OK();
}

static Status TryHandleHardSigmoidSequence(std::vector<const NodeUnit*>& fused_nodes,
                                           QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& start_node_unit,
                                           const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                           const logging::Logger& logger,
                                           bool do_op_validation) {
  // Looking for a standalone HardSigmoid to start the sequence.
  if (start_node_unit.OpType() != "HardSigmoid" || start_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& hs_node = start_node_unit.GetNode();

  // HardSigmoid must have a single Mul child. HardSigmoid must not produce a graph output.
  auto children = graph_utils::FindChildrenByType(hs_node, "Mul");
  if (children.size() != 1 || hs_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(hs_node)) {
    return Status::OK();
  }

  const Node& mul_node = *children[0];
  const auto mul_node_unit_it = node_unit_map.find(&mul_node);

  ORT_RETURN_IF(mul_node_unit_it == node_unit_map.end(), "Node does not have a corresponding NodeUnit");

  const NodeUnit* mul_node_unit = mul_node_unit_it->second;

  // Mul child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (mul_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();  // THIS would be an invalid model.
  }

  // Input to HardSigmoid must also be the other input to the Mul.
  auto& hs_input_name = start_node_unit.Inputs()[0].node_arg.Name();

  bool same_root_input = false;
  for (const auto& mul_input_def : mul_node_unit->Inputs()) {
    if (mul_input_def.node_arg.Name() == hs_input_name) {
      same_root_input = true;
      break;
    }
  }

  if (!same_root_input) {
    return Status::OK();
  }

  // TODO: Check HardSigmoid alpha and beta values.
  // TODO: Add a HardSwish to model.
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);

  fused_nodes.push_back(&start_node_unit);
  fused_nodes.push_back(mul_node_unit);

  return Status::OK();
}

using FusionFunc = Status (*)(std::vector<const NodeUnit*>&,
                              QnnModelWrapper&,
                              const NodeUnit&,
                              const std::unordered_map<const Node*, const NodeUnit*>&,
                              const logging::Logger&,
                              bool);

Status TryFusions(/*out*/ std::vector<const NodeUnit*>& fused_nodes,
                  QnnModelWrapper& qnn_model_wrapper,
                  const NodeUnit& starting_node,
                  const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                  const logging::Logger& logger,
                  bool validate) {
  ORT_RETURN_IF_NOT(fused_nodes.empty(), "fused_nodes is not empty");

  std::array<FusionFunc, 2> fusions = {
      TryHandleConvertSequence,
      TryHandleHardSigmoidSequence,
  };

  for (auto fusion : fusions) {
    ORT_RETURN_IF_ERROR(fusion(fused_nodes, qnn_model_wrapper, starting_node, node_unit_map, logger, validate));
    if (!fused_nodes.empty()) {
      return Status::OK();
    }
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
