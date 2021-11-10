// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(cudaStream_t stream, output_t* grad_input, const input_t* grad, const input_t* output, int softmax_elements, int softmax_elements_stride, int batch_count);

template <typename T>
class SoftmaxGrad final : public CudaKernel {
 public:
  SoftmaxGrad(const OpKernelInfo& info) : CudaKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();
    axis_ = info.GetAttrOrDefault("axis", static_cast<int64_t>(opset_ < 13 ? 1 : -1));
    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmaxGrad";
    cuda_ep_ = const_cast<CUDAExecutionProvider*>(
        static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool log_softmax_;
  int opset_;
  CUDAExecutionProvider* cuda_ep_;
};

}  // namespace cuda
}  // namespace onnxruntime
