// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace ort_dnnl {
template <typename T>
class Gemm final : public OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
