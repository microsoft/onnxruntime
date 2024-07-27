// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_fusions.h"

namespace onnxruntime {
namespace qnn {

Status QnnConvActivationFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                                  gsl::span<const NodeUnit*> dq_node_units,
                                  const NodeUnit* conv_node_unit,
                                  const NodeUnit* activation_node_unit,
                                  const NodeUnit* q_node_unit,
                                  const logging::Logger& logger,
                                  bool validate = false);

std::optional<QnnNodeGroup> TryConvActivationFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& conv_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger);
}  // namespace qnn
}  // namespace onnxruntime
