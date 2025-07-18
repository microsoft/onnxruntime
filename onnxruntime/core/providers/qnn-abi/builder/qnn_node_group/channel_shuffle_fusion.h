// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of pattern:  Transpose -> ChannelShuffle (Reshape -> Transpose -> Reshape) -> Transpose
/// </summary>
class ChannelShuffleFusion : public IQnnNodeGroup {
 public:
  explicit ChannelShuffleFusion(gsl::span<const OrtNodeUnit* const> node_units) {
    ORT_ENFORCE(node_units.size() == 5, "Pattern expect exactly 5 NodeUnits.");
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
    node_units_[3] = node_units[3];
    node_units_[4] = node_units[4];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ChannelShuffleFusion);

  Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "ChannelShuffleFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a channel shuffle pattern.
  /// If so, returns a IQnnNodeGroup that contains the ChannelShuffle NodeUnits.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& transpose_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 5> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
