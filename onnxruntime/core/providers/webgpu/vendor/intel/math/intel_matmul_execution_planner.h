// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/math/matmul_execution_planner.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

class IntelMatMulExecutionPlanner final : public MatMulExecutionPlanner {
 public:
  IntelMatMulExecutionPlanner() = default;
  explicit IntelMatMulExecutionPlanner(SplitKConfig split_k_config);

 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& params) const override;
};

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
