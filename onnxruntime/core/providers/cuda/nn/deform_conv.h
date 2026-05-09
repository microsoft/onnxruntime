// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/deform_conv_attributes.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class DeformConv final : public CudaKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : CudaKernel(info), attrs_(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  DeformConvAttributes attrs_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeformConv);
};

}  // namespace cuda
}  // namespace onnxruntime
