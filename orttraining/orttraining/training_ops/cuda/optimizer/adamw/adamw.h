// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "orttraining/training_ops/cpu/optimizer/adamw/adamwbase.h"

namespace onnxruntime {
namespace cuda {

class AdamWOptimizer final : public CudaKernel, public contrib::AdamWOptimizerBase {
 public:
  AdamWOptimizer(const OpKernelInfo& info) : CudaKernel(info), contrib::AdamWOptimizerBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
