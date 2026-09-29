// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

class IntelMatMulAlgorithmScheduler final : public MatMulAlgorithmScheduler {
 public:
  IntelMatMulAlgorithmScheduler() = default;
  explicit IntelMatMulAlgorithmScheduler(SplitKConfig split_k_config);

 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& params) const override;
};

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
