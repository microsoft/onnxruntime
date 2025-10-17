// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of pattern: Reshape -> Transpose -> Reshape where intermediate tensors are rank-6.
/// QNN doesn't support rank-6 Reshape and Transpose operators, so this fusion converts them to rank-5
/// by removing a unit dimension (value of 1) from intermediate tensors.
/// Pattern: Tensor(t0) -> Reshape(R1) -> Tensor(t1) -> Transpose(T1) -> Tensor(t2) -> Reshape(R2) -> Tensor(t3)
/// Conditions:
/// - Rank(t0) == Rank(t3) AND Last dimension of t0 equals last dimension of t3
/// - Rank(t1) == Rank(t2) == 6
/// - There exists a dimension index where both t1 and t2 have value 1
/// - Transpose must leave that unit dimension in place (perm[unit_dim_index] == unit_dim_index)
/// </summary>
class Rank6ToRank5Fusion : public IQnnNodeGroup {
 public:
  explicit Rank6ToRank5Fusion(gsl::span<const NodeUnit* const> node_units, size_t unit_dim_index)
      : unit_dim_index_(unit_dim_index) {
    ORT_ENFORCE(node_units.size() == 3, "Pattern expects exactly 3 NodeUnits.");
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Rank6ToRank5Fusion);

  Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "Rank6ToRank5Fusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Reshape -> Transpose -> Reshape
  /// pattern with rank-6 intermediate tensors.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& reshape1_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 3> node_units_;  // Reshape1, Transpose, Reshape2
  size_t unit_dim_index_;                      // Index of the unit dimension (value 1) to remove
};

}  // namespace qnn
}  // namespace onnxruntime
