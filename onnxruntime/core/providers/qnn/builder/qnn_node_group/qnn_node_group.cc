// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group.h"

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/conv_activation_fusion.h"
#include "core/providers/qnn/builder/qnn_node_group/dq_q_fusion.h"
#include "core/providers/qnn/builder/qnn_node_group/hardsigmoid_mul_fusion.h"

namespace onnxruntime {
namespace qnn {

std::string_view QnnNodeGroup::TypeToString(QnnNodeGroup::Type type) {
  static std::array<std::string_view, static_cast<size_t>(QnnNodeGroup::Type::COUNT)> type_names = {
      "Undefined",
      "NodeUnit",
      "ConvActivationFusion",
      "DQQFusion",
      "HardSigmoidMulFusion",
  };

  return type_names[static_cast<size_t>(type)];
}

Status QnnNodeGroup::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  using Func = Status (*)(
      QnnModelWrapper&,
      const QnnNodeGroup&,
      const logging::Logger&);

  static std::array<Func, static_cast<size_t>(QnnNodeGroup::Type::COUNT)> funcs = {
      [](QnnModelWrapper&, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) -> Status {
        std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(qnn_node_group.type_),
                                           " in QnnNodeGroup::IsSupported()");
        LOGS(logger, ERROR) << error_msg;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
      },
      [](QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) -> Status {
        ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 1 && qnn_node_group.node_units_[0] != nullptr, "");
        const NodeUnit& node_unit = *qnn_node_group.node_units_[0];
        const std::string& op_type = node_unit.OpType();
        const auto* op_builder = qnn::GetOpBuilder(op_type);

        if (op_builder == nullptr) {
          std::string err_msg = MakeString("Operators of type `", op_type,
                                           "` are not supported by QNN EP.", op_type, " node `",
                                           node_unit.Name(), "` will not be assigned to QNN EP.");
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, err_msg);
        }

        Status status = op_builder->IsOpSupported(qmw, *qnn_node_group.node_units_[0], logger);
        if (!status.IsOK()) {
          LOGS(logger, WARNING) << op_type << " node `" << node_unit.Name()
                                << "` is not supported: " << status.ErrorMessage();
        }

        return status;
      },
      conv_act_fusion::IsSupported,
      dq_q_fusion::IsSupported,
      hs_mul_fusion::IsSupported,
  };

  return funcs[static_cast<size_t>(type_)](qmw, *this, logger);
}

Status QnnNodeGroup::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  using Func = Status (*)(
      QnnModelWrapper&,
      const QnnNodeGroup&,
      const logging::Logger&);

  static std::array<Func, static_cast<size_t>(QnnNodeGroup::Type::COUNT)> funcs = {
      [](QnnModelWrapper&, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) -> Status {
        std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(qnn_node_group.type_),
                                           " in QnnNodeGroup::AddToModelBuilder()");
        LOGS(logger, ERROR) << error_msg;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
      },
      [](QnnModelWrapper& qmw, const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) -> Status {
        ORT_RETURN_IF_NOT(qnn_node_group.node_units_.size() == 1 && qnn_node_group.node_units_[0] != nullptr, "");
        const auto* op_builder = qnn::GetOpBuilder(qnn_node_group.node_units_[0]->OpType());
        ORT_RETURN_IF_NOT(op_builder != nullptr, "[QNN EP]: Missing OpBuilder for OpType ", qnn_node_group.node_units_[0]->OpType());
        return op_builder->AddToModelBuilder(qmw, *qnn_node_group.node_units_[0], logger, /*do_op_validation*/ false);
      },
      conv_act_fusion::AddToModelBuilder,
      dq_q_fusion::AddToModelBuilder,
      hs_mul_fusion::AddToModelBuilder,
  };

  return funcs[static_cast<size_t>(type_)](qmw, *this, logger);
}

const NodeUnit* QnnNodeGroup::GetTargetNodeUnit(const logging::Logger& logger) const {
  using Func = const NodeUnit* (*)(const QnnNodeGroup&, const logging::Logger&);

  static std::array<Func, static_cast<size_t>(QnnNodeGroup::Type::COUNT)> funcs = {
      [](const QnnNodeGroup& qnn_node_group, const logging::Logger& logger) -> const NodeUnit* {
        std::string error_msg = MakeString("Unhandled QnnNodeGroup::Type ", TypeToString(qnn_node_group.type_),
                                           " in QnnNodeGroup::AddToModelBuilder()");
        LOGS(logger, ERROR) << error_msg;
        return nullptr;
      },
      [](const QnnNodeGroup& qnn_node_group, const logging::Logger&) -> const NodeUnit* {
        if (qnn_node_group.node_units_.size() != 1) {
          return nullptr;
        }
        return qnn_node_group.node_units_[0];
      },
      conv_act_fusion::GetTargetNodeUnit,
      dq_q_fusion::GetTargetNodeUnit,
      hs_mul_fusion::GetTargetNodeUnit,
  };

  return funcs[static_cast<size_t>(type_)](*this, logger);
}

using FusionFunc = std::optional<QnnNodeGroup> (*)(
    QnnModelWrapper&,
    const NodeUnit&,
    const std::unordered_map<const Node*, const NodeUnit*>&,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>&,
    const logging::Logger&);

static std::optional<QnnNodeGroup> TryQnnFusions(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& starting_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // Maps a starting operator type to the fusion function.
  static std::unordered_map<std::string, FusionFunc> fusions = {
      {"DequantizeLinear", TryDQQFusion},
      {"HardSigmoid", TryHardSigmoidMulFusion},
      {"Conv", TryConvActivationFusion},
      {"ConvTranspose", TryConvActivationFusion},
  };

  // For now, all fusions involve standalone node units (i.e., no wrapping DQ/Q nodes).
  if (starting_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  auto iter = fusions.find(starting_node_unit.OpType());
  if (iter != fusions.end()) {
    FusionFunc fusion_func = iter->second;
    return fusion_func(qnn_model_wrapper, starting_node_unit, node_to_node_unit,
                       node_unit_to_qnn_node_group, logger);
  }
  return std::nullopt;
}

Status GetQnnNodeGroups(/*out*/ std::vector<QnnNodeGroup>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                        const size_t num_node_units,
                        const logging::Logger& logger) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::vector<NodeIndex> sorted_node_indices = graph_viewer.GetNodesInTopologicalOrder();

  std::vector<QnnNodeGroup::IndexType> sorted_qnn_node_group_indices;
  sorted_qnn_node_group_indices.reserve(num_node_units);

  std::vector<QnnNodeGroup> tmp_qnn_node_groups;
  tmp_qnn_node_groups.reserve(num_node_units);

  {
    std::unordered_map<const NodeUnit*, QnnNodeGroup::IndexType> node_unit_to_qnn_node_group;
    std::vector<gsl::not_null<const NodeUnit*>> sorted_node_units;
    sorted_node_units.reserve(num_node_units);

    // Create QnnNodeGroups for fusions first.
    for (NodeIndex node_index : sorted_node_indices) {
      gsl::not_null<const Node*> node = graph_viewer.GetNode(node_index);

      // Get the NodeUnit associated with the node.
      const auto node_unit_it = node_to_node_unit.find(node);
      ORT_RETURN_IF_NOT(node_unit_it != node_to_node_unit.end(), "Could not find NodeUnit for Node ", node->Name());
      gsl::not_null<const NodeUnit*> node_unit = node_unit_it->second;

      // Skip this node if it is not the NodeUnit's target node to ensure NodeUnits are visited in topological order.
      if (node != &node_unit->GetNode()) {
        continue;
      }

      sorted_node_units.push_back(node_unit);

      if (node_unit_to_qnn_node_group.count(node_unit) != 0) {
        continue;  // Already handled this node unit
      }

      std::optional<QnnNodeGroup> fused_node_group = TryQnnFusions(qnn_model_wrapper, *node_unit,
                                                                   node_to_node_unit, node_unit_to_qnn_node_group,
                                                                   logger);

      if (fused_node_group.has_value()) {
        const QnnNodeGroup::IndexType index = tmp_qnn_node_groups.size();
        fused_node_group->index_ = index;

        for (const NodeUnit* fused_node_unit : fused_node_group->GetNodeUnits()) {
          assert(fused_node_unit != nullptr);
          node_unit_to_qnn_node_group.insert({fused_node_unit, index});
        }

        tmp_qnn_node_groups.push_back(std::move(*fused_node_group));
      }
    }

    // Create QnnNodeGroups for the leftover NodeUnits.
    for (gsl::not_null<const NodeUnit*> node_unit : sorted_node_units) {
      const auto it = node_unit_to_qnn_node_group.find(node_unit);
      if (it != node_unit_to_qnn_node_group.end()) {
        // Already handled this NodeUnit.
        const QnnNodeGroup& qnn_node_group = tmp_qnn_node_groups[it->second];
        if (node_unit == qnn_node_group.GetTargetNodeUnit(logger)) {
          sorted_qnn_node_group_indices.push_back(qnn_node_group.index_);
        }
        continue;
      }

      const QnnNodeGroup::IndexType index = tmp_qnn_node_groups.size();
      QnnNodeGroup fused_node_group = {};
      fused_node_group.type_ = QnnNodeGroup::Type::NodeUnit;
      fused_node_group.index_ = index;
      fused_node_group.node_units_.resize(1);
      fused_node_group.node_units_[0] = node_unit;
      tmp_qnn_node_groups.push_back(std::move(fused_node_group));

      node_unit_to_qnn_node_group.insert({node_unit, index});
      sorted_qnn_node_group_indices.push_back(index);
    }

    assert(tmp_qnn_node_groups.size() == sorted_qnn_node_group_indices.size());
  }

  // Copy QnnNodeGroups to output in sorted (topological) order.
  qnn_node_groups.resize(0);
  qnn_node_groups.reserve(tmp_qnn_node_groups.size());
  for (auto index : sorted_qnn_node_group_indices) {
    assert(index < tmp_qnn_node_groups.size());
    QnnNodeGroup qnn_node_group = std::move(tmp_qnn_node_groups[index]);
    qnn_node_group.index_ = qnn_node_groups.size();
    qnn_node_groups.push_back(std::move(qnn_node_group));
  }

  assert(qnn_node_groups.size() == sorted_qnn_node_group_indices.size());

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
