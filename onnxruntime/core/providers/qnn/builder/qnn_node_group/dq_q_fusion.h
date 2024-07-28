// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

class DQQFusion : public IQnnNodeGroup {
 public:
  DQQFusion(const NodeUnit& dq_node_unit, const NodeUnit& q_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DQQFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "DQQFusion"; }

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
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& dq_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 2> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
