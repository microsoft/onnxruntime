// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {
class CastLoneQFusion : public IQnnNodeGroup {
 public:
  explicit CastLoneQFusion(gsl::span<const NodeUnit* const> node_units) {
    ORT_ENFORCE(node_units.size() == 2, "Pattern expect exactly 2 NodeUnits.");
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(CastLoneQFusion);

  Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "CastLoneQFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Cast -> Quantize sequence.
  /// If so, returns a IQnnNodeGroup that contains the Cast and Quantize NodeUnits.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& mul_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 2> node_units_;
  bool IsIntToFloatCast(const NodeUnit& node_unit) const;
};

}  // namespace qnn
}  // namespace onnxruntime