// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

struct QnnNodeGroup {
  using IndexType = size_t;
  enum class Type : uint8_t {
    Undefined = 0,
    NodeUnit,
    ConvActivationFusion,
    DQQFusion,
    HardSigmoidMulFusion,
    COUNT,
  };

  static std::string_view TypeToString(QnnNodeGroup::Type type);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const;
  const std::vector<const NodeUnit*>& GetNodeUnits() const { return node_units_; }
  const NodeUnit* GetTargetNodeUnit(const logging::Logger& logger) const;

  QnnNodeGroup::Type type_ = QnnNodeGroup::Type::Undefined;
  IndexType index_ = 0;
  std::vector<const NodeUnit*> node_units_;
};

Status GetQnnNodeGroups(/*out*/ std::vector<QnnNodeGroup>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                        size_t num_node_units,
                        const logging::Logger& logger);
}  // namespace qnn
}  // namespace onnxruntime
