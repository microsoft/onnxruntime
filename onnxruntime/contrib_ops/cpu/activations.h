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
    const T* input = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    int num_threads = concurrency::ThreadPool::NumThreads(tp);
    int64_t elem_count = X->Shape().Size();
    if (elem_count > num_threads) {
      concurrency::ThreadPool::TryBatchParallelFor(
          tp, elem_count, [input, output, elem_count, num_threads](std::ptrdiff_t i) {
            int64_t elem_inx_start = i * elem_count / num_threads;
            int64_t elem_inx_end = (i + 1) * elem_count / num_threads;
            for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
              output[elem_inx] = input[elem_inx] * static_cast<float>(M_SQRT1_2);
            }
            MlasComputeErf(output + elem_inx_start, output + elem_inx_start, elem_inx_end - elem_inx_start);
            for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
              output[elem_inx] = 0.5f * input[elem_inx] * (output[elem_inx] + 1.0f);
            }
          },
          num_threads);
      return Status::OK();
    }

    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = xm * static_cast<float>(M_SQRT1_2);
    MlasComputeErf(Y->template MutableData<T>(), Y->template MutableData<T>(), X->Shape().Size());
    ym = xm * 0.5f * (ym + 1.0f);
    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
