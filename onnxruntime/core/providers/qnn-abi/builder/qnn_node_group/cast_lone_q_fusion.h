// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {
/// <summary>
/// Represents a fusion of pattern: Quantize(Cast(x)) => Convert(x)
/// when x is not the output of Dequantize
/// </summary>
class CastLoneQFusion : public IQnnNodeGroup {
 public:
  explicit CastLoneQFusion(gsl::span<const OrtNodeUnit* const> node_units) {
    if (node_units.size() != 2) {
      ORT_CXX_API_THROW("Pattern expect exactly 2 OrtNodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(CastLoneQFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "CastLoneQFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Cast -> Quantize sequence.
  /// If so, returns a IQnnNodeGroup that contains the Cast and Quantize NodeUnits.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& mul_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
