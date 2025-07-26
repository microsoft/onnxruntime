// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a {DQ, DQ->Q->DQ} -> Gemm -> Q sequence.
/// This is translated into a QNN's FC w/ LPBQ encodings.
/// The contained NodeUnits are of type SingleNode since they are not part of a QDQ node unit.
/// </summary>

class LowPowerBlockQuantizedGemmFusion : public IQnnNodeGroup {
 public:
  LowPowerBlockQuantizedGemmFusion(const NodeUnit& Scale_DQL_node_unit,
                                   const NodeUnit& W_QL_node_unit,
                                   const NodeUnit& W_DQL_node_unit,
                                   const NodeUnit& Act_DQL_node_unit,
                                   const NodeUnit& Gemm_node_unit,
                                   const NodeUnit& Output_QL_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LowPowerBlockQuantizedGemmFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "LowPowerBlockQuantizedGemmFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& scale_dql_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::array<const NodeUnit*, 6> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime