// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;
/// <summary>
/// Represents a fusion of a DQ - UDO - Q
/// This is translated into a QNN custom op. One day this should be implemented in the QDQ actions.
/// The contained NodeUnits are of type SingleNode since they are not a part of a QDQ node unit.
/// </summary>
class UDOQDQFusion : public IQnnNodeGroup {
 public:
  UDOQDQFusion(
      const std::string& op_type,
      const std::string& op_package,
      const std::map<size_t, const NodeUnit*>& input_dq_units,
      const NodeUnit& node_unit,
      const std::map<size_t, const NodeUnit*>& output_q_units);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(UDOQDQFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "UDOQDQFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid sequence.
  /// If so, returns a IQnnNodeGroup that contains the UDO, 2x DQ and Q NodeUnits.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and traverse/query the graph</param>
  /// <param name="input_dq_unit">DQ  node unit that could start the sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      const std::string& op_type,
      const std::string& op_package,
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& input_dq_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  const std::string op_type_;
  const std::string op_package_;
  const std::map<size_t, const NodeUnit*> input_node_units_;
  const NodeUnit* node_unit_;
  const std::map<size_t, const NodeUnit*> output_node_units_;
  std::vector<const NodeUnit*> all_nodes_;
};

}  // namespace qnn
}  // namespace onnxruntime
