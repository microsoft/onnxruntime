// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

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
std::optional<QnnNodeGroup> TryHardSigmoidMulFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& hardsigmoid_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger);

namespace hs_mul_fusion {

Status IsSupported(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
Status AddToModelBuilder(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
// const std::vector<const NodeUnit*>& GetNodeUnits(const QnnNodeGroup& qnn_node_group);
const NodeUnit* GetTargetNodeUnit(const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
}  // namespace hs_mul_fusion
}  // namespace qnn
}  // namespace onnxruntime
