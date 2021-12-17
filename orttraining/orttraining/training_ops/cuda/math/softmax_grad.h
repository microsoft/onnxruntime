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
  SoftmaxGrad(const OpKernelInfo& info) : CudaKernel{info},
                                          prop_(static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp()) {
    const auto& node = info.node();
    opset_ = (node.OpType() == "SoftmaxGrad_13" || node.OpType() == "LogSoftmaxGrad_13") ? 13 : 1;
    axis_ = info.GetAttrOrDefault("axis", static_cast<int64_t>(opset_ < 13 ? 1 : -1));
    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmaxGrad" || info.GetKernelDef().OpName() == "LogSoftmaxGrad_13";
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool log_softmax_;
  int opset_;  // opset_ of the forward Softmax/LogSoftmax operator
  const cudaDeviceProp& prop_;
};

}  // namespace cuda
}  // namespace onnxruntime
