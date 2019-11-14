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
    if (nullptr == tp) {
      EIGEN_X_VAR(xm);
      EIGEN_Y_VAR(ym);
      ym = xm * static_cast<float>(M_SQRT1_2);
      MlasComputeErf(Y->template MutableData<T>(), Y->template MutableData<T>(), X->Shape().Size());
      ym = xm * 0.5f * (ym + 1.0f);
      return Status::OK();
    } else {
      const T* input = X->template Data<T>();
      T* output = Y->template MutableData<T>();
      int thread_num = tp->NumThreads();
      int64_t elem_size = X->Shape().Size();
      int64_t elem_per_thread = elem_size / thread_num;
      int64_t elem_residual = elem_size % thread_num;
      tp->ParallelFor(thread_num, [input,
                                   output,
                                   elem_size,
                                   elem_per_thread,
                                   elem_residual](int32_t i) {
        int64_t elem_this_thread = i < elem_residual ? elem_per_thread + 1 : elem_per_thread;
        int64_t elem_inx_start = i * elem_per_thread + (i < elem_residual ? i : elem_residual);
        for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_start + elem_this_thread; elem_inx++) {
          output[elem_inx] = input[elem_inx] * static_cast<float>(M_SQRT1_2);
        }
        MlasComputeErf(output + elem_inx_start, output + elem_inx_start, elem_this_thread);
        for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_start + elem_this_thread; elem_inx++) {
          output[elem_inx] = 0.5f * input[elem_inx] * (output[elem_inx] + 1.0f);
        }
      });
      return Status::OK();
    }
  }
};

}  // namespace contrib
}  // namespace onnxruntime
