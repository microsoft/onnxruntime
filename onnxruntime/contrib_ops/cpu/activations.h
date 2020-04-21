// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>

namespace onnxruntime {
namespace contrib {

template <typename T>
class ScaledTanh final : public OpKernel {
 public:
  ScaledTanh(const OpKernelInfo& info)
      : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)), beta_(info.GetAttrOrDefault("beta", 1.0f)) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = (T)alpha_ * (EIGEN_X * (T)beta_).tanh();
    return Status::OK();
  }

 private:
  const float alpha_;
  const float beta_;
};

template <typename T>
class Gelu : public OpKernel {
 public:
  Gelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    //The cost comes from microbenchmark(manual tunning).
    const double cost = 10.0;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    concurrency::ThreadPool::TryParallelFor(tp, batch_size, cost, [data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output + first, len);
      ym = xm * static_cast<float>(M_SQRT1_2);
      MlasComputeErf(output, output, len);
      ym = xm * 0.5f * (ym + 1.0f);
    });
    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
