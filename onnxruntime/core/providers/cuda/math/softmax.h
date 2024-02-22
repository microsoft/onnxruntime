// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TOut, bool is_log_softmax>
Status SoftMaxComputeHelper(
    Stream* stream,
    const T* input,
    const TensorShape& shape,
    TOut* Y,
    int64_t axis);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_warpwise_softmax_forward(Stream* stream, output_t* dst, const input_t* src,
                                         int softmax_elements, int softmax_elements_stride, int batch_count);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_blockwise_softmax_forward(Stream* stream, output_t* output, const input_t* input,
                                          int softmax_elements, int input_stride, int output_stride, int batch_count);

template <typename T>
class Softmax final : public CudaKernel {
 public:
  Softmax(const OpKernelInfo& info) : CudaKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }

    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool log_softmax_;
  int opset_;
};

}  // namespace cuda
}  // namespace onnxruntime
