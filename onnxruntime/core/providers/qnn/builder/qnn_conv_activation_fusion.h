// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

Status QnnConvActivationFusionAdd(QnnModelWrapper& qnn_model_wrapper,
                                  gsl::span<const NodeUnit*> dq_node_units,
                                  const NodeUnit* conv_node_unit,
                                  const NodeUnit* activation_node_unit,
                                  const NodeUnit* q_node_unit,
                                  const logging::Logger& logger,
                                  bool validate = false);

Status TryConvActivationFusion(/*out*/ std::vector<const NodeUnit*>& fused_nodes,
                               QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& conv_node_unit,
                               const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                               const std::unordered_set<const NodeUnit*>& handled_node_units,
                               const logging::Logger& logger,
                               bool do_op_validation);
}  // namespace qnn
}  // namespace onnxruntime
