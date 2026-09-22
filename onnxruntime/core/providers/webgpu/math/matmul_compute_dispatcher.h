// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"

namespace onnxruntime {

class Tensor;

namespace webgpu {

struct Activation;
class ComputeContext;
class ComputeContextBase;

class SubgroupMatrixMatMulImpl {
 public:
  virtual ~SubgroupMatrixMatMulImpl() = default;

  virtual bool CanApply(const ComputeContext& context,
                        const std::vector<const Tensor*>& inputs,
                        bool is_channels_last,
                        bool b_is_constant) const = 0;

  virtual Status Compute(ComputeContext& context,
                         const std::vector<const Tensor*>& inputs,
                         Tensor* output,
                         const Activation& activation,
                         bool is_channels_last,
                         bool b_is_constant) = 0;
};

class MatMulComputeDispatcher {
 public:
  MatMulComputeDispatcher() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulComputeDispatcher);

  Status Compute(ComputeContext& context,
                 const Activation& activation,
                 const std::vector<const Tensor*>& inputs,
                 Tensor* output,
                 bool is_channels_last,
                 bool b_is_constant = false);

 private:
  void Initialize(const ComputeContextBase& context);

  std::once_flag init_flag_;
  std::unique_ptr<SubgroupMatrixMatMulImpl> subgroup_matrix_impl_;
  std::unique_ptr<MatMulAlgorithmScheduler> scheduler_;
};

}  // namespace webgpu
}  // namespace onnxruntime
