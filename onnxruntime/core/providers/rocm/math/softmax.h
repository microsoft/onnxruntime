// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

using tunable::RocmTuningContext;

template <typename T, typename TOut, bool IsLogSoftmax>
Status SoftMaxComputeHelper(
    Stream* stream,
    const T* input,
    const TensorShape& shape,
    TOut* Y,
    int64_t axis,
    RocmTuningContext* tuning_ctx = nullptr);

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_warpwise_softmax_forward(Stream* stream, OutputT* dst, const InputT* src, int softmax_elements,
                                         int softmax_elements_stride, int batch_count,
                                         RocmTuningContext* tuning_ctx = nullptr);

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_blockwise_softmax_forward(Stream* stream, OutputT* output, const InputT* input, int softmax_elements,
                                          int input_stride, int output_stride, int batch_count,
                                          RocmTuningContext* tuning_ctx = nullptr);

template <typename T>
class Softmax final : public RocmKernel {
 public:
  Softmax(const OpKernelInfo& info) : RocmKernel{info} {
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

}  // namespace rocm
}  // namespace onnxruntime
