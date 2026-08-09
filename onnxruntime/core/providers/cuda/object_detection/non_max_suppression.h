// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"

namespace onnxruntime {
namespace cuda {

struct NonMaxSuppression final : public CudaKernel, public NonMaxSuppressionBaseImpl<OpKernelInfo, OpKernelContext> {
  explicit NonMaxSuppression(const OpKernelInfo& info) : CudaKernel(info), NonMaxSuppressionBaseImpl(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NonMaxSuppression);
};
}  // namespace cuda
}  // namespace onnxruntime
