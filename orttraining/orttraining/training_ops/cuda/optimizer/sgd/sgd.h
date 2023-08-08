// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "orttraining/training_ops/cpu/optimizer/sgd/sgdbase.h"

namespace onnxruntime {
namespace cuda {

class SGDOptimizerV2 final : public CudaKernel, public contrib::SGDOptimizerV2Base {
 public:
  SGDOptimizerV2(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
