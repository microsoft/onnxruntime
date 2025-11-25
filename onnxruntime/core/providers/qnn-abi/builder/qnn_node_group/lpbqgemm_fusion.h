// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a {DQ, DQ->Q->DQ} -> Gemm -> Q sequence.
/// This is translated into a QNN's FC w/ LPBQ encodings.
/// The contained OrtNodeUnits are of type SingleNode since they are not part of a QDQ node unit.
/// </summary>

class LowPowerBlockQuantizedGemmFusion : public IQnnNodeGroup {
 public:
  LowPowerBlockQuantizedGemmFusion(const OrtNodeUnit& Scale_DQL_node_unit,
                                   const OrtNodeUnit& W_QL_node_unit,
                                   const OrtNodeUnit& W_DQL_node_unit,
                                   const OrtNodeUnit& Act_DQL_node_unit,
                                   const OrtNodeUnit& Gemm_node_unit,
                                   const OrtNodeUnit& Output_QL_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LowPowerBlockQuantizedGemmFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "LowPowerBlockQuantizedGemmFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& scale_dql_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 6> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
