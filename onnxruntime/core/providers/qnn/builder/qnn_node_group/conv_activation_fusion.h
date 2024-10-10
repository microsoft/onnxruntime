// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a DQ* -> Conv -> Relu/Clip -> Q sequence where the Relu (or Clip) is redundant
/// due to the quantization effects of the Q. This sequence is translated to a quantized QNN Conv.
/// All contained NodeUnits are of type SingleNode since they are not a part of an existing QDQ node unit.
/// </summary>
class ConvActivationFusion : public IQnnNodeGroup {
 public:
  ConvActivationFusion(const NodeUnit& dq_node_unit_0,
                       const NodeUnit& dq_node_unit_1,
                       const NodeUnit* dq_node_unit_2,
                       const NodeUnit& conv_node_unit,
                       const NodeUnit& activation_node_unit,
                       const NodeUnit& q_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ConvActivationFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ConvActivationFusion"; }

  /// <summary>
  /// Traverses graph to check if the given NodeUnit is part of a valid DQ* -> Conv -> Relu/Clip -> Q sequence.
  /// If so, returns a IQnnNodeGroup that contains the constituent NodeUnits.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and to traverse/query the graph</param>
  /// <param name="conv_node_unit">Conv node unit (type SingleNode) that be part of the sequence.</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& conv_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 6> node_units_;  // Last elem is nullptr if the optional bias DQ is missing.
};

}  // namespace qnn
}  // namespace onnxruntime
