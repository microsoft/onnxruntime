// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of pattern: Softmax(Mul(x, scalar_scale)) => QnnSoftmax(x, beta=scalar_scale)
/// </summary>
class ScaleSoftmaxFusion : public IQnnNodeGroup {
 public:
  ScaleSoftmaxFusion(const OrtNodeUnit& mul_node_unit, const OrtNodeUnit& softmax_node_unit)
      : node_units_{&mul_node_unit, &softmax_node_unit} {
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ScaleSoftmaxFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[1]; }
  std::string_view Type() const override { return "ScaleSoftmaxFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Mul -> Softmax sequence.
  /// If so, returns a IQnnNodeGroup that contains the Mul and Softmax NodeUnits.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& mul_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
