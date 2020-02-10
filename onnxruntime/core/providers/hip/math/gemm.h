// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class Gemm final : public HipKernel {
  using Base = HipKernel;

 public:
  Gemm(const OpKernelInfo& info) : HipKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
};

}  // namespace hip
}  // namespace onnxruntime
