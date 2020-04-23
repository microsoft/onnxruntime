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
    const Tensor* input = context->Input<Tensor>(0);
    const T* input_data = input->template Data<T>();

    Tensor* output = context->Output(0, input->Shape());
    T* output_data = output->template MutableData<T>();

    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    int64_t elem_count = input->Shape().Size();
    // FastGelu allows optional bias. Here we split input data into chunks. Each chunk
    // has N elements (except the last chunk), and use thread pool to parallel chunks.
    // N = 4096 is selected based on performance test results on input shape 1x128x768.
    static const int64_t length_per_task = 4096;
    int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          for (int64_t i = 0; i < count; i++) {
            T value = p_input[i];
            p_output[i] = value * static_cast<float>(M_SQRT1_2);
          }

          MlasComputeErf(p_output, p_output, count);

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
    return Status::OK();
  }
};  // namespace contrib

}  // namespace contrib
}  // namespace onnxruntime
