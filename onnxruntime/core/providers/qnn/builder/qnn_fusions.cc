// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_fusions.h"

#include <limits>
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
static Status TryHandleConvertSequence(std::vector<const NodeUnit*>& fused_nodes,
                                       QnnModelWrapper& qnn_model_wrapper,
                                       const NodeUnit& dq_node_unit,
                                       const NodeUnit& q_node_unit,
                                       const logging::Logger& logger,
                                       bool do_op_validation) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Looking for a standalone DQ to start the sequence.
  assert(dq_node_unit.OpType() == QDQ::DQOpName && dq_node_unit.UnitType() == NodeUnit::Type::SingleNode);
  assert(q_node_unit.OpType() == QDQ::QOpName && q_node_unit.UnitType() == NodeUnit::Type::SingleNode);

  const Node& dq_node = dq_node_unit.GetNode();
  const Node& q_node = q_node_unit.GetNode();

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  // DQ and Q must have equal scale type and different zp type.
  if (!QDQ::IsDQQConversion(dq_node, q_node, get_const_initializer, graph_viewer.ModelPath())) {
    return Status::OK();
  }

  const auto& node_name = utils::GetNodeName(dq_node_unit);
  const NodeUnitIODef& input_def = dq_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = q_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  // Run QNN validation on the final fused node before committing to doing a fusion.
  // Importantly, this validation process does not modify the qnn_model_wrapper.
  // If validation fails here, we return Status::OK() to allow QNN EP to use the normal OpBuilder workflow.
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor), logger);
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor), logger);
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                           QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                           QNN_OP_CONVERT,
                                                           {input_tensor.GetQnnTensor()},
                                                           {output_tensor.GetQnnTensor()},
                                                           {}),
                         logger);

  // Validation passed, so we're now committed to doing a fusion. The following statements modify qnn_model_wrapper.
  // If we encounter an error, we return it directly to caller.
  LOGS(logger, VERBOSE) << " Adding QNN Convert via fusion. dq_node name: [" << dq_node.Name()
                        << "] dq_node optype: [" << dq_node.OpType()
                        << "] q_node name: [" << q_node_unit.Name()
                        << "] q_node optype: [" << q_node_unit.OpType()
                        << "]";

  // Add a QNN Convert to the model. Get the input from the DQ node, and the output from the Q node.
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(q_node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CONVERT,
                                                    {input_def.node_arg.Name()},
                                                    {output_def.node_arg.Name()},
                                                    {},
                                                    do_op_validation),
                    "Failed to add fused Convert node.");

  fused_nodes.push_back(&dq_node_unit);
  fused_nodes.push_back(&q_node_unit);

  return Status::OK();
}

/**
 * Tries to fuse sequences that start with a DQ node.
 *
 * \param fused_nodes Output list of node units that were fused. Remains empty if fusion is not applied.
 * \param qnn_model_wrapper The QNN model that is being built.
 * \param dq_node_unit The DQ node unit that could potentially start a sequence.
 * \param node_unit_map Maps a node to its node unit.
 * \param handled_node_units Set of node units that have already been processed. Fusion will not fuse nodes
 *                           in this set.
 * \param logger The logger.
 * \param do_op_validation True if should call QNN operator validation APIs.
 * \return An onnxruntime::Status
 */
static Status TryHandleDequantize(std::vector<const NodeUnit*>& fused_nodes,
                                  QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& dq_node_unit,
                                  const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                  const std::unordered_set<const NodeUnit*>& handled_node_units,
                                  const logging::Logger& logger,
                                  bool do_op_validation) {
  // Expect that this function is called with a standalone DQ.
  assert(dq_node_unit.OpType() == QDQ::DQOpName && dq_node_unit.UnitType() == NodeUnit::Type::SingleNode);

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& dq_node = dq_node_unit.GetNode();

  // DQ must have a single child (1 output edge) and must not produce a graph output.
  if (dq_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(dq_node)) {
    return Status::OK();
  }

  const Node& child_node = dq_node.OutputEdgesBegin()->GetNode();
  const auto child_node_unit_it = node_unit_map.find(&child_node);
  ORT_RETURN_IF(child_node_unit_it == node_unit_map.end(), "Node does not have a corresponding NodeUnit");
  const NodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if this
  // fusion function has been called in topological order, but check to be safe.
  if (handled_node_units.count(child_node_unit) != 0) {
    return Status::OK();
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  const std::string& child_type = child_node.OpType();

  // Try (DQ -> Q) into QNN's Convert op.
  if (child_type == QDQ::QOpName) {
    return TryHandleConvertSequence(fused_nodes, qnn_model_wrapper, dq_node_unit, *child_node_unit,
                                    logger, do_op_validation);
  }

  // Try (DQ -> Conv/ConvTranspose -> Relu/Clip -> Q) into QNN Conv/ConvTranspose.
  if (child_type == "Conv" || child_type == "ConvTranspose") {
    return TryConvActivationFusion(fused_nodes, qnn_model_wrapper, *child_node_unit, node_unit_map,
                                   handled_node_units, logger, do_op_validation);
  }

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
static Status TryHandleHardSigmoidSequence(std::vector<const NodeUnit*>& fused_nodes,
                                           QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& start_node_unit,
                                           const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                           const std::unordered_set<const NodeUnit*>& handled_node_units,
                                           const logging::Logger& logger,
                                           bool do_op_validation) {
  // Looking for a standalone HardSigmoid to start the sequence.
  if (start_node_unit.OpType() != "HardSigmoid" || start_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  NodeAttrHelper hs_attr_helper(start_node_unit);
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

  // Check if Mul node has already been handled. Should not be the case if this
  // fusion function has been called in topological order, but check to be safe.
  if (handled_node_units.count(mul_node_unit) != 0) {
    return Status::OK();
  }

  // Mul child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (mul_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  // Input to HardSigmoid must also be the other input to the Mul.
  auto& hs_input_name = start_node_unit.Inputs()[0].node_arg.Name();
  const bool same_root_input = mul_node.InputDefs()[0]->Name() == hs_input_name ||
                               mul_node.InputDefs()[1]->Name() == hs_input_name;

  if (!same_root_input) {
    return Status::OK();
  }

  const auto& node_name = utils::GetNodeName(start_node_unit);
  const NodeUnitIODef& input_def = start_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = mul_node_unit->Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  // Run QNN validation on the final fused node before committing to doing a fusion.
  // Importantly, this validation process does not modify the qnn_model_wrapper.
  // If validation fails here, we return Status::OK() to allow QNN EP to use the normal OpBuilder workflow.
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor), logger);
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor), logger);
  QNN_RETURN_OK_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                           QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                           QNN_OP_HARD_SWISH,
                                                           {input_tensor.GetQnnTensor()},
                                                           {output_tensor.GetQnnTensor()},
                                                           {}),
                         logger);

  // Validation passed, so we're now committed to doing a fusion. The following statements modify qnn_model_wrapper.
  // If we encounter an error, we return it directly to caller.
  LOGS(logger, VERBOSE) << " Adding QNN HardSwish via fusion. HardSigmoid name: [" << start_node_unit.Name()
                        << "] Mul name: [" << mul_node_unit->Name() << "]";

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_HARD_SWISH,
                                                    {input_def.node_arg.Name()},
                                                    {output_def.node_arg.Name()},
                                                    {},
                                                    do_op_validation),
                    "Failed to add fused HardSwish node.");

  fused_nodes.push_back(&start_node_unit);
  fused_nodes.push_back(mul_node_unit);

  return Status::OK();
}

using FusionFunc = Status (*)(std::vector<const NodeUnit*>&,
                              QnnModelWrapper&,
                              const NodeUnit&,
                              const std::unordered_map<const Node*, const NodeUnit*>&,
                              const std::unordered_set<const NodeUnit*>&,
                              const logging::Logger&,
                              bool);

Status TryFusions(/*out*/ std::vector<const NodeUnit*>& fused_nodes,
                  QnnModelWrapper& qnn_model_wrapper,
                  const NodeUnit& starting_node,
                  const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                  const std::unordered_set<const NodeUnit*>& handled_node_units,
                  const logging::Logger& logger,
                  bool validate) {
  // Maps a starting operator type to the fusion function.
  static std::unordered_map<std::string, FusionFunc> fusions = {
      {"DequantizeLinear", TryHandleDequantize},
      {"HardSigmoid", TryHandleHardSigmoidSequence},
  };

  // For now, all fusions involve standalone node units (i.e., no wrapping DQ/Q nodes).
  if (starting_node.UnitType() != NodeUnit::Type::SingleNode) {
    return Status::OK();
  }

  auto iter = fusions.find(starting_node.OpType());
  if (iter != fusions.end()) {
    fused_nodes.clear();

    FusionFunc fusion_func = iter->second;
    ORT_RETURN_IF_ERROR(fusion_func(fused_nodes, qnn_model_wrapper, starting_node, node_unit_map,
                                    handled_node_units, logger, validate));
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
