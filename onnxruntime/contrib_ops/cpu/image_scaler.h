// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib{

template <typename T>
class ImageScaler final : public OpKernel {
 public:
  ImageScaler(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());
    ORT_ENFORCE(info.GetAttrs<float>("bias", bias_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    const auto& dims = X->Shape().GetDims();

    if (dims.size() < 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input is expected to have four dimensions corresponding to [N,C,H,W], got ", dims.size());
    }

    const int64_t N = dims[0];
    const int64_t C = dims[1];
    const int64_t H = dims[2];
    const int64_t W = dims[3];

    if (!bias_.empty() && bias_.size() != static_cast<size_t>(C)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Bias size (", bias_.size(), ") does not match the number of channels (", C, ")");
    }

    Tensor* Y = context->Output(0, TensorShape({N, C, H, W}));
    ConstEigenArrayMap<T> X_arr(X->template Data<T>(), H * W, N * C);
    EigenArrayMap<T> Y_arr(Y->template MutableData<T>(), H * W, N * C);

    for (int64_t nc = 0; nc < N * C; ++nc) {
      Y_arr.col(nc) = scale_ * X_arr.col(nc) + bias_[nc % C];
    }
    return Status::OK();
  }

 protected:
  float scale_;
  std::vector<float> bias_;
};
}
}  //namespace onnxruntime
