// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/object_detection/non_max_suppression.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

class NonMaxSuppression final : public CudaKernel, public NonMaxSuppressionBase {
 public:
  explicit NonMaxSuppression(const OpKernelInfo& info) : 
    CudaKernel(info), NonMaxSuppressionBase(info)
  {}
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime