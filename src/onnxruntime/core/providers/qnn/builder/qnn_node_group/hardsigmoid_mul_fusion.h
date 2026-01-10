// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a HardSigmoid -> Mul sequence that computes `x * HardSigmoid<alpha=1/6, beta=0.5>(x)`.
/// This is translated into a QNN HardSwish operator.
/// The contained NodeUnits are of type SingleNode since they are not a part of a QDQ node unit.
/// </summary>
class HardSigmoidMulFusion : public IQnnNodeGroup {
 public:
  HardSigmoidMulFusion(const NodeUnit& hardsigmoid_node_unit, const NodeUnit& mul_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(HardSigmoidMulFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "HardSigmoidMulFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid HardSigmoid -> Mul sequence.
  /// If so, returns a IQnnNodeGroup that contains the HardSigmoid and Mul NodeUnits.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and traverse/query the graph</param>
  /// <param name="hardsigmoid_node_unit">HardSigmoid node unit that could start the sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& hardsigmoid_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 2> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
