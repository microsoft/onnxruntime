// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_fusions.h"

#include <vector>
#include "core/graph/graph_utils.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

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

  std::array<FusionFunc, 1> fusions = {
      TryHandleConvertSequence,
  };

  for (auto fusion : fusions) {
    ORT_RETURN_IF_ERROR(fusion(fused_nodes, qnn_model_wrapper, starting_node, node_unit_map, logger, validate));
    if (!fused_nodes.empty()) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status TryHandleHardSigmoidSequence(std::vector<const NodeUnit*>& fused_nodes,
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

}  // namespace qnn
}  // namespace onnxruntime
