// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/split.h"
#include "orttraining/training_ops/cpu/tensor/split.h"

namespace onnxruntime {
namespace cuda {

class SplitTraining final : public CudaKernel, public SplitBase {
 public:
  SplitTraining(const OpKernelInfo& info) : CudaKernel(info), SplitBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
