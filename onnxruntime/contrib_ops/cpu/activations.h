// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/activation/element_wise_ranged_transform.h"

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

template <typename T>
class Gelu : public OpKernel {
 public:
  Gelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input = context->Input<Tensor>(0);
    const T* input_data = input->template Data<T>();

    Tensor* output = context->Output(0, input->Shape());
    T* output_data = output->template MutableData<T>();

    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    int64_t elem_count = input->Shape().Size();
    static const int64_t length_per_task = 4096;  // this number comes from FastGelu.
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
            p_output[i] = value * static_cast<T>(M_SQRT1_2);
          }

          MlasComputeErf(p_output, p_output, count);

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
