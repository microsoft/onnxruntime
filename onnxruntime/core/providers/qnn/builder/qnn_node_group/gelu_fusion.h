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
/// Represents a fusion of the Gelu pattern expanded into ONNX operators.
/// This fusion handles two patterns:
///   Pattern 1:
///                +-------Mul(0.5)---------------------+
///                |                                    |
///                |                                    v
///             [root] --> Div -----> Erf  --> Add --> Mul ==>
///                       (B=1.4142...)        (1)
///
///   Pattern 2:
///                +------------------------------------+
///                |                                    |
///                |                                    v
///             [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
///                       (B=1.4142...)        (1)            (0.5)
///
/// Both patterns are translated into a QNN Gelu operator.
/// The contained NodeUnits can be of type SingleNode or QDQGroup (with Q-DQ nodes).
/// The second inputs to Div, Add, and Mul Node Units should be constant.
/// </summary>
class GeluFusion : public IQnnNodeGroup {
 public:
  GeluFusion(std::vector<const NodeUnit*>&& node_units, const NodeUnit* target_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(GeluFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "GeluFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Gelu pattern.
  /// If so, returns a IQnnNodeGroup that contains all the NodeUnits in the pattern.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and traverse/query the graph</param>
  /// <param name="erf_node_unit">Erf node unit that could be part of the sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& erf_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::vector<const NodeUnit*> node_units_;
  const NodeUnit* target_node_unit_;
};

}  // namespace qnn
}  // namespace onnxruntime
