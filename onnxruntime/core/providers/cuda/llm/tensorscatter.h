// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class TensorScatter final : public CudaKernel {
 public:
  TensorScatter(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool circular_;
};

}  // namespace cuda
}  // namespace onnxruntime
