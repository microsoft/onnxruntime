// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <gsl/gsl>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

std::unique_ptr<IQnnNodeGroup> TryConvActivationFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& conv_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger);

namespace conv_act_fusion {

class QnnNodeGroup : public IQnnNodeGroup {
 public:
  QnnNodeGroup(gsl::span<const NodeUnit*> dq_node_units,
               const NodeUnit& conv_node_unit,
               const NodeUnit& activation_node_unit,
               const NodeUnit& q_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnNodeGroup);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  std::vector<const NodeUnit*> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ConvActivationFusion"; }

 private:
  std::array<const NodeUnit*, 3> dq_node_units_;  // Last DQ is nullptr if bias is missing.
  const NodeUnit& conv_node_unit_;
  const NodeUnit& activation_node_unit_;
  const NodeUnit& q_node_unit_;
};

}  // namespace conv_act_fusion
}  // namespace qnn
}  // namespace onnxruntime
