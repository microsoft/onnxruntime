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
    if (nullptr != tp) {
      const T* input = X->template Data<T>();
      T* output = Y->template MutableData<T>();
      int task_count = tp->NumThreads() + 1;
      int64_t elem_count = X->Shape().Size();
      if (elem_count > task_count) {
        tp->ParallelFor(task_count, [input,
                                     output,
                                     elem_count,
                                     task_count](int32_t i) {
          int64_t elem_inx_start = i * elem_count / task_count;
          int64_t elem_inx_end = (i + 1) * elem_count / task_count;
          for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
            output[elem_inx] = input[elem_inx] * static_cast<float>(M_SQRT1_2);
          }
          MlasComputeErf(output + elem_inx_start, output + elem_inx_start, elem_inx_end - elem_inx_start);
          for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
            output[elem_inx] = 0.5f * input[elem_inx] * (output[elem_inx] + 1.0f);
          }
        });
        return Status::OK();
      }
    }

    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = xm * static_cast<float>(M_SQRT1_2);
    MlasComputeErf(Y->template MutableData<T>(), Y->template MutableData<T>(), X->Shape().Size());
    ym = xm * 0.5f * (ym + 1.0f);
    return Status::OK();
  }
};

template <typename T>
class FastGelu : public OpKernel {
 public:
  FastGelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    if (nullptr != tp) {
      const T* input = X->template Data<T>();
      T* output = Y->template MutableData<T>();
      int task_count = tp->NumThreads() + 1;
      int64_t elem_count = X->Shape().Size();
      const auto coefficient = kAlpha * kGamma;
      if (elem_count > task_count) {
        tp->ParallelFor(task_count, [ input,
                                      output,
                                      elem_count,
                                      task_count,
                                      kAlpha = this->kAlpha,
                                      coefficient ](int32_t i) {
          int64_t elem_inx_start = i * elem_count / task_count;
          int64_t elem_inx_end = (i + 1) * elem_count / task_count;
          for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
            const auto x = input[elem_inx];
            output[elem_inx] = x * (coefficient * x * x  + kAlpha);
            output[elem_inx] = 0.5f * x * (1.0f + tanh(output[elem_inx]));
          }
        });
        return Status::OK();
      }
    }

    //
    // Commented out EIGEN implentation due to EIGEN bug.
    // On Windows build with GPU enabled, kGamma * x_pow_3 + EIGEN_X below would produce incorrect
    // result, same issue discovered in fast_gelu_grad op.
    // Given that CPU kernel is mostly for conformance check, where performance is not of high
    // priority, to workaround this bug, use a for loop and avoid using EIGEN library.
    //
    // const auto x_pow_3 = EIGEN_X.cube();
    // const auto tanh_result = (kAlpha * (kGamma * x_pow_3 + EIGEN_X)).tanh();

    // EIGEN_Y = 0.5f * EIGEN_X * (1.f + tanh_result);

    const T* input = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    int64_t elem_count = X->Shape().Size();
    for (auto i = 0; i < elem_count; ++i) {
      const auto x_val = input[i];
      const auto x_cube = x_val * x_val * x_val;
      T tanh_result = std::tanh(kAlpha * x_val + kAlpha * kGamma * x_cube);
      output[i] = 0.5f * x_val * (tanh_result + 1.0f);
    }
    return Status::OK();
  }

 private:
  static constexpr float kAlpha = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);
  static constexpr float kGamma = 0.044715f;
};

}  // namespace contrib
}  // namespace onnxruntime
