// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class LayerNormCudnn final : public CudaKernel {
 public:
  LayerNormCudnn(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

template <typename T>
class LayerNormCudnnGrad final : public CudaKernel {
 public:
  LayerNormCudnnGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  cudnnReduceTensorDescriptor_t reduce_sum_desc_{nullptr};
  cudnnReduceTensorDescriptor_t reduce_mean_desc_{nullptr};
};

}  // namespace cuda
}  // namespace onnxruntime
