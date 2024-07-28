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
std::optional<QnnNodeGroup> TryDQQFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& dq_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger);

namespace dq_q_fusion {

Status IsSupported(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
Status AddToModelBuilder(QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
// const std::vector<const NodeUnit*>& GetNodeUnits(const QnnNodeGroup& qnn_node_group);
const NodeUnit* GetTargetNodeUnit(const QnnNodeGroup& qnn_node_group, const logging::Logger& logger);
}  // namespace dq_q_fusion
}  // namespace qnn
}  // namespace onnxruntime
