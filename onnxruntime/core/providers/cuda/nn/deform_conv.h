// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cpu/nn/deform_conv_attributes.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

struct DeformConvState {
  TensorShape last_x_dims;
  TensorShape last_w_dims;
  int cached_n_parallel_imgs{0};
  std::mutex mutex;
};

template <typename T>
class DeformConv final : public CudaKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : CudaKernel(info), attrs_(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status UpdateState(OpKernelContext* context,
                     const DeformConvParams& params,
                     int& n_parallel_imgs) const;

  DeformConvAttributes attrs_;
  mutable DeformConvState state_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeformConv);
};

}  // namespace cuda
}  // namespace onnxruntime
