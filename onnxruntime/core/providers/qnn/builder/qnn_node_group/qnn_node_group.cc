// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group.h"

#include <gsl/gsl>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
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

class QnnNodeUnitWrapper : public IQnnNodeGroup {
 public:
  QnnNodeUnitWrapper(const NodeUnit& node_unit) : node_unit_(&node_unit) {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnNodeUnitWrapper);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override {
    const std::string& op_type = node_unit_->OpType();
    const auto* op_builder = qnn::GetOpBuilder(op_type);
    ORT_RETURN_IF_NOT(op_builder != nullptr, "Operators of type `", op_type,
                      "` are not supported by QNN EP.", op_type, " node `",
                      node_unit_->Name(), "` will not be assigned to QNN EP.");

    return op_builder->IsOpSupported(qmw, *node_unit_, logger);
  }

  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override {
    const std::string& op_type = node_unit_->OpType();
    const auto* op_builder = qnn::GetOpBuilder(op_type);
    ORT_RETURN_IF_NOT(op_builder != nullptr, "[QNN EP]: Missing OpBuilder for OpType ", op_type);
    return op_builder->AddToModelBuilder(qmw, *node_unit_, logger, /*do_op_validation*/ false);
  }

  gsl::span<const NodeUnit* const> GetNodeUnits() const override {
    return gsl::span<const NodeUnit* const>{&node_unit_, 1ULL};
  }

  const NodeUnit* GetTargetNodeUnit() const override { return node_unit_; }
  std::string_view Type() const override { return "NodeUnitWrapper"; }

 private:
  const NodeUnit* node_unit_;
};

using FusionFunc = std::unique_ptr<IQnnNodeGroup> (*)(
    QnnModelWrapper&,
    const NodeUnit&,
    const std::unordered_map<const Node*, const NodeUnit*>&,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>&,
    const logging::Logger&);

static std::unique_ptr<IQnnNodeGroup> TryQnnFusions(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& starting_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
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
    return nullptr;
  }

  auto iter = fusions.find(starting_node_unit.OpType());
  if (iter != fusions.end()) {
    FusionFunc fusion_func = iter->second;
    return fusion_func(qnn_model_wrapper, starting_node_unit, node_to_node_unit,
                       node_unit_to_qnn_node_group, logger);
  }
  return nullptr;
}

Status GetQnnNodeGroups(/*out*/ std::vector<std::unique_ptr<IQnnNodeGroup>>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                        const size_t num_node_units,
                        const logging::Logger& logger) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::vector<NodeIndex> sorted_node_indices = graph_viewer.GetNodesInTopologicalOrder();

  std::vector<size_t> sorted_qnn_node_group_indices;
  sorted_qnn_node_group_indices.reserve(num_node_units);

  std::vector<std::unique_ptr<IQnnNodeGroup>> tmp_qnn_node_groups;
  tmp_qnn_node_groups.reserve(num_node_units);

  {
    std::unordered_map<const NodeUnit*, const IQnnNodeGroup*> node_unit_to_qnn_node_group;
    std::unordered_map<const IQnnNodeGroup*, size_t> fused_qnn_node_group_indices;
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

      std::unique_ptr<IQnnNodeGroup> fused_node_group = TryQnnFusions(qnn_model_wrapper, *node_unit,
                                                                      node_to_node_unit, node_unit_to_qnn_node_group,
                                                                      logger);

      if (fused_node_group) {
        const size_t index = tmp_qnn_node_groups.size();
        fused_qnn_node_group_indices[fused_node_group.get()] = index;

        for (const NodeUnit* fused_node_unit : fused_node_group->GetNodeUnits()) {
          assert(fused_node_unit != nullptr);
          node_unit_to_qnn_node_group.insert({fused_node_unit, fused_node_group.get()});
        }

        tmp_qnn_node_groups.push_back(std::move(fused_node_group));
      }
    }

    // Create QnnNodeGroups for the leftover NodeUnits.
    for (gsl::not_null<const NodeUnit*> node_unit : sorted_node_units) {
      const auto it = node_unit_to_qnn_node_group.find(node_unit);
      if (it != node_unit_to_qnn_node_group.end()) {
        // Already handled this NodeUnit.
        gsl::not_null<const IQnnNodeGroup*> fused_qnn_node_group = it->second;
        if (node_unit == fused_qnn_node_group->GetTargetNodeUnit()) {
          sorted_qnn_node_group_indices.push_back(fused_qnn_node_group_indices[fused_qnn_node_group]);
        }
        continue;
      }

      const size_t index = tmp_qnn_node_groups.size();
      auto qnn_node_group = std::make_unique<QnnNodeUnitWrapper>(*node_unit);

      node_unit_to_qnn_node_group.insert({node_unit, qnn_node_group.get()});
      tmp_qnn_node_groups.push_back(std::move(qnn_node_group));
      sorted_qnn_node_group_indices.push_back(index);
    }

    assert(tmp_qnn_node_groups.size() == sorted_qnn_node_group_indices.size());
  }

  // Copy QnnNodeGroups to output in sorted (topological) order.
  qnn_node_groups.resize(0);
  qnn_node_groups.reserve(tmp_qnn_node_groups.size());
  for (auto index : sorted_qnn_node_group_indices) {
    assert(index < tmp_qnn_node_groups.size());
    std::unique_ptr<IQnnNodeGroup> qnn_node_group = std::move(tmp_qnn_node_groups[index]);
    qnn_node_groups.push_back(std::move(qnn_node_group));
  }

  assert(qnn_node_groups.size() == sorted_qnn_node_group_indices.size());

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
