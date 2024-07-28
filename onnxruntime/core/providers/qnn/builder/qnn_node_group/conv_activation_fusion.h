// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <gsl/gsl>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

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

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& conv_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 6> node_units_;  // Last elem is nullptr if bias DQ is missing.
};

}  // namespace qnn
}  // namespace onnxruntime
