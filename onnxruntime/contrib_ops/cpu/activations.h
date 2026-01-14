// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"
using onnxruntime::narrow;
namespace onnxruntime {
namespace functors {

template <typename T>
struct ScaledTanh : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 5.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = alpha * (xm * beta).tanh();
  }
};
template <typename T>
struct ParametricSoftplus : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 15.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (T)alpha *
         (xm * (T)beta > 0)
             .select(xm * (T)beta + ((-xm * (T)beta).exp() + 1.0f).log(), ((xm * (T)beta).exp() + 1.0f).log());
  }
};
}  // namespace functors

namespace contrib {
DEFINE_ELE_KERNEL(ScaledTanh);
DEFINE_ELE_KERNEL(ParametricSoftplus);

// Implement a new one instead of inheriting from ElementWiseRangedTransform so that we can call
// MlasComputeLogistic instead of using Eigen for better perf.
template <typename T>
class QuickGelu : public OpKernel {
 public:
  QuickGelu(const OpKernelInfo& info) : OpKernel(info) { alpha_ = info.GetAttrOrDefault<float>("alpha", 1.702f); }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input = context->Input<Tensor>(0);
    const T* input_data = input->template Data<T>();
    Tensor* output = context->Output(0, input->Shape());
    T* output_data = output->template MutableData<T>();
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    int64_t elem_count = input->Shape().Size();
    constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
    int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          if (alpha_ != 1.0f) {
            // TODO: Consider vectorizing this scalar multiplication.
            // It needs exposing a new API in MLAS to take in a scalar
            // that will be used in the elementwise multiplication.
            // Estimate the cost-benefit tradeoff before proceeding
            // with that optimization.
            for (int64_t i = 0; i < count; i++) {
              p_output[i] = p_input[i] * alpha_;
            }

            MlasComputeLogistic(p_output, p_output, onnxruntime::narrow<size_t>(count));
          } else {
            // SILU activation - this needs no `alpha_` scaling as `alpha_` will be 1.0f
            MlasComputeLogistic(p_input, p_output, onnxruntime::narrow<size_t>(count));
          }

          MlasEltwiseMul<float>(p_input, p_output, p_output, onnxruntime::narrow<size_t>(count));
        },
        0);

    return Status::OK();
  }

 private:
  float alpha_;
};

}  // namespace contrib
}  // namespace onnxruntime
