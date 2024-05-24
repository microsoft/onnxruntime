// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/tensor/gelu.h"

using onnxruntime::narrow;
using namespace onnxruntime::common;

namespace onnxruntime {

// May revisit the implementations to support inplace computation, if needed.

ONNX_CPU_OPERATOR_KERNEL(
    Gelu,
    20,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gelu<float>);

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gelu<float>);
}
#endif

template <typename T>
Status Gelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->Data<T>();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  if (approximation_algorithm_ == "tanh") {
    // FastGelu allows optional bias. Here we split input data into chunks. Each chunk
    // has N elements (except the last chunk), and use thread pool to parallel chunks.
    // N = 4096 is selected based on performance test results on input shape 1x128x768.
    // FastGelu uses approximation for Gelu. The formula is 0.5 * (1 + Tanh(x * (C * x * x + B))) * x.
    static constexpr float B = 0.7978845608028654f;    // sqrt(2.0 / M_PI)
    static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0 / M_PI)

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          for (int64_t i = 0; i < count; i++) {
            T value = p_input[i];
            p_output[i] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
          }

          MlasComputeTanh(p_output, p_output, narrow<size_t>(count));

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
    return Status::OK();
  } else if (approximation_algorithm_ == "none") {
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

          MlasComputeErf(p_output, p_output, narrow<size_t>(count));

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}

}  // namespace onnxruntime
