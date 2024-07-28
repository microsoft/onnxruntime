// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
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
std::unique_ptr<IQnnNodeGroup> TryHardSigmoidMulFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& hardsigmoid_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger);

namespace hs_mul_fusion {

class QnnNodeGroup : public IQnnNodeGroup {
 public:
  QnnNodeGroup(const NodeUnit& hardsigmoid_node_unit, const NodeUnit& mul_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnNodeGroup);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  std::vector<const NodeUnit*> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "HardSigmoidMulFusion"; }

 private:
  const NodeUnit& hardsigmoid_node_unit_;
  const NodeUnit& mul_node_unit_;
};

}  // namespace hs_mul_fusion
}  // namespace qnn
}  // namespace onnxruntime
