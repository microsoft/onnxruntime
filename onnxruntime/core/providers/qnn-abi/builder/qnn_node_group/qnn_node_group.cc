// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
// #include "core/providers/qnn-abi/builder/qnn_node_group/dq_q_fusion.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/hardsigmoid_mul_fusion.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
// #include "core/providers/qnn-abi/builder/qnn_node_group/reshape_gemm_fusion.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

/// <summary>
/// A IQnnNodeGroup class that wraps a single NodeUnit. Most NodeUnits in the ONNX graph will
/// be wrapped by this class.
/// </summary>
class QnnNodeUnitWrapper : public IQnnNodeGroup {
 public:
  explicit QnnNodeUnitWrapper(const OrtNodeUnit& node_unit) : node_unit_(&node_unit) {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnNodeUnitWrapper);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override {
    const std::string& op_type = node_unit_->OpType();
    const auto* op_builder = qnn::GetOpBuilder(op_type);

    const std::string& name = node_unit_->Name();
    ORT_RETURN_IF_NOT(op_builder != nullptr, "Operators of type `", op_type,
                      "` are not supported by QNN EP.", op_type, " node `",
                      name, "` will not be assigned to QNN EP.");

    return op_builder->IsOpSupported(qmw, *node_unit_, logger);
  }

  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override {
    const std::string& op_type = node_unit_->OpType();
    const auto* op_builder = qnn::GetOpBuilder(op_type);
    ORT_RETURN_IF_NOT(op_builder != nullptr, "[QNN EP]: Missing OpBuilder for OpType ", op_type);
    return op_builder->AddToModelBuilder(qmw, *node_unit_, logger, /*do_op_validation*/ false);
  }

  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override {
    return gsl::span<const OrtNodeUnit* const>{&node_unit_, 1ULL};
  }

  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_unit_; }
  std::string_view Type() const override { return "OrtNodeUnit"; }

 private:
  const OrtNodeUnit* node_unit_;
};

/// <summary>
/// The type of a function that tries to fuse NodeUnits into a IQnnNodeGroup.
/// </summary>
using FusionFunc = std::unique_ptr<IQnnNodeGroup> (*)(
    QnnModelWrapper&,
    const OrtNodeUnit&,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>&,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>&,
    const logging::Logger&);

// / <summary>
// / Given a starting NodeUnit, this function tries all possible fusions that start with that NodeUnit.
// / If successful, returns a IQnnNodeGroup object that represents the fusion of various NodeUnits.
// / Currently only handles standalone NodeUnits that are not in a QDQ unit but that can change in the future.
// / </summary>
// / <param name="qnn_model_wrapper">QnnModelWrapper that contains the ONNX GraphViewer. Used for validation.</param>
// / <param name="starting_node_unit">NodeUnit that potentially starts a fusion.</param>
// / <param name="node_to_node_unit">Maps a Node* to a NodeUnit*</param>
// / <param name="node_unit_to_qnn_node_group">Maps a NodeUnit* to a IQnnNodeGroup*</param>
// / <param name="logger"></param>
// / <returns>IQnnNodeGroup representing the fusion or an empty std::unique_ptr</returns>
static std::unique_ptr<IQnnNodeGroup> TryQnnFusions(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& starting_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // Maps a starting operator type to the fusion function.
  static std::unordered_map<std::string, FusionFunc> fusions = {
    //   {"DequantizeLinear", DQQFusion::TryFusion},
    //   {"HardSigmoid", HardSigmoidMulFusion::TryFusion},
    //   {"Gemm", ReshapeGemmFusion::TryFusion},
    //   {"Mul", ScaleSoftmaxFusion::TryFusion},
    //   {"Transpose", ChannelShuffleFusion::TryFusion}
    };

  // For now, all fusions involve standalone node units (i.e., no wrapping DQ/Q nodes).
  if (starting_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
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

// Traverses the ONNX Graph and groups NodeUnits into IQnnNodeGroup objects. Some IQnnNodeGroup objects
// represent a fusion of various NodeUnits. This function generates a vector of indices that
// represent the topological order of the qnn_node_groups.
static Status GetQnnNodeGroupsImpl(/*out*/ std::vector<std::unique_ptr<IQnnNodeGroup>>& qnn_node_groups,
                                   /*out*/ std::vector<size_t>& sorted_qnn_node_group_indices,
                                   QnnModelWrapper& qnn_model_wrapper,
                                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
                                   const size_t num_node_units,
                                   const logging::Logger& logger) {
  const OrtGraph& graph = qnn_model_wrapper.GetOrtGraph();
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  OrtArrayOfConstObjects* nodes = nullptr;
  ort_api.Graph_GetNodes(&graph, &nodes);

  size_t num_nodes = 0;
  ort_api.ArrayOfConstObjects_GetSize(nodes, &num_nodes);

  const void* const* node_data = nullptr;
  ort_api.ArrayOfConstObjects_GetData(nodes, &node_data);

  sorted_qnn_node_group_indices.reserve(num_node_units);
  qnn_node_groups.reserve(num_node_units);

  std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*> node_unit_to_qnn_node_group;
  std::unordered_map<const IQnnNodeGroup*, size_t> fused_qnn_node_group_indices;
  std::vector<gsl::not_null<const OrtNodeUnit*>> sorted_node_units;
  sorted_node_units.reserve(num_node_units);

  // Process just the fusions of NodeUnits first to ensure a correct topological order of all IQnnNodeGroups.
  // This is the same approach taken by ORT utilities for grouping Nodes into NodeUnits.
  for (size_t node_index = 0; node_index < num_nodes; ++node_index) {
    gsl::not_null<const OrtNode*> node = static_cast<const OrtNode*>(node_data[node_index]);

    // Get the NodeUnit associated with the node.
    const auto node_unit_it = node_to_node_unit.find(node);
    ORT_RETURN_IF_NOT(node_unit_it != node_to_node_unit.end(), "Could not find NodeUnit for Node.");
    gsl::not_null<const OrtNodeUnit*> node_unit = node_unit_it->second;

    // Skip this node if it is not the NodeUnit's target node to ensure NodeUnits are visited in topological order.
    if (node != &node_unit->GetNode()) {
      continue;
    }

    sorted_node_units.push_back(node_unit);

    if (node_unit_to_qnn_node_group.count(node_unit) != 0) {
      continue;  // Already handled this node unit
    }

    logger;
    std::unique_ptr<IQnnNodeGroup> fused_node_group = TryQnnFusions(qnn_model_wrapper, *node_unit,
                                                                    node_to_node_unit, node_unit_to_qnn_node_group,
                                                                    logger);

    // if (fused_node_group) {
    //   const size_t index = qnn_node_groups.size();
    //   fused_qnn_node_group_indices[fused_node_group.get()] = index;

    //   for (const NodeUnit* fused_node_unit : fused_node_group->GetNodeUnits()) {
    //     assert(fused_node_unit != nullptr);
    //     node_unit_to_qnn_node_group.insert({fused_node_unit, fused_node_group.get()});
    //   }

    //   qnn_node_groups.push_back(std::move(fused_node_group));
    // }
  }

  // Create IQnnNodeGroups for the leftover NodeUnits that were not fused.
  for (gsl::not_null<const OrtNodeUnit*> node_unit : sorted_node_units) {
    const auto it = node_unit_to_qnn_node_group.find(node_unit);

    if (it != node_unit_to_qnn_node_group.end()) {
      // Already added this NodeUnit to a IQnnNodeGroup, so we'll skip it.
      // However, if this NodeUnit is the "target" for the IQnnNodeGroup, then add its index to
      // the sorted list of indices.
      gsl::not_null<const IQnnNodeGroup*> fused_qnn_node_group = it->second;
      if (node_unit == fused_qnn_node_group->GetTargetNodeUnit()) {
        sorted_qnn_node_group_indices.push_back(fused_qnn_node_group_indices[fused_qnn_node_group]);
      }
      continue;
    }

    const size_t index = qnn_node_groups.size();
    auto qnn_node_group = std::make_unique<QnnNodeUnitWrapper>(*node_unit);

    node_unit_to_qnn_node_group.insert({node_unit, qnn_node_group.get()});
    qnn_node_groups.push_back(std::move(qnn_node_group));
    sorted_qnn_node_group_indices.push_back(index);
  }

  assert(qnn_node_groups.size() == sorted_qnn_node_group_indices.size());

  return Status::OK();
}

Status GetQnnNodeGroups(/*out*/ std::vector<std::unique_ptr<IQnnNodeGroup>>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
                        const size_t num_node_units,
                        const logging::Logger& logger) {
  std::vector<size_t> sorted_qnn_node_group_indices;
  std::vector<std::unique_ptr<IQnnNodeGroup>> qnn_node_groups_holder;
  ORT_RETURN_IF_ERROR(GetQnnNodeGroupsImpl(qnn_node_groups_holder, sorted_qnn_node_group_indices, qnn_model_wrapper,
                                           node_to_node_unit, num_node_units, logger));

  // Move IQnnNodeGroups to the output std::vector in sorted (topological) order.
  qnn_node_groups.resize(0);
  qnn_node_groups.reserve(qnn_node_groups_holder.size());
  for (auto index : sorted_qnn_node_group_indices) {
    assert(index < qnn_node_groups_holder.size());
    std::unique_ptr<IQnnNodeGroup> qnn_node_group = std::move(qnn_node_groups_holder[index]);
    qnn_node_groups.push_back(std::move(qnn_node_group));
  }

  assert(qnn_node_groups.size() == sorted_qnn_node_group_indices.size());

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
