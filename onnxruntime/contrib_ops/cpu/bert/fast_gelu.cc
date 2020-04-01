// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fast_gelu.h"

#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "fast_gelu_helper.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    FastGelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FastGelu<float>);

// FastGelu = 0.5 * (1 + Tanh(x * (C * x * x + B))) * X
static constexpr float B = 0.7978845608028654f;    // sqrt(2.0 / M_PI)
static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0 / M_PI)

template <typename T>
Status FastGelu<T>::ComputeGelu(OpKernelContext* context, const T* input_data, T* output_data, int64_t elem_count) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  if (nullptr != tp) {
    int task_count = tp->NumThreads();
    if (elem_count > task_count) {
      tp->SimpleParallelFor(task_count, [input_data,
                                         output_data,
                                         elem_count,
                                         task_count](std::ptrdiff_t task_idx) {
        int64_t start = task_idx * elem_count / task_count;
        int64_t end = (task_idx + 1) * elem_count / task_count;
        for (int64_t index = start; index < end; index++) {
          T value = input_data[index];
          output_data[index] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
        }
        MlasComputeTanh(output_data + start, output_data + start, end - start);
        for (int64_t index = start; index < end; index++) {
          output_data[index] = 0.5f * input_data[index] * (output_data[index] + 1.0f);
        }
      });
      return Status::OK();
    }
  }

  ConstEigenVectorArrayMap<T> xm(input_data, elem_count);
  EigenVectorArrayMap<T> ym(output_data, elem_count);
  ym = xm * (static_cast<T>(C) * xm * xm + static_cast<T>(B));
  MlasComputeTanh(output_data, output_data, elem_count);
  ym = xm * 0.5f * (ym + 1.0f);
  return Status::OK();
}

template <typename T>
Status FastGelu<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(fast_gelu::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->template Data<T>();
  int64_t elem_count = input->Shape().Size();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->template MutableData<T>();

  const Tensor* bias = context->Input<Tensor>(1);
  if (nullptr == bias) {
    return ComputeGelu(context, input_data, output_data, elem_count);
  }
  const T* bias_data = bias->template Data<T>();
  int64_t bias_len = bias->Shape().Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr buffer = BufferUniquePtr(alloc->Alloc(SafeInt<size_t>(sizeof(T)) * elem_count),
                                           BufferDeleter(alloc));
  T* tmp_data = static_cast<T*>(buffer.get());

  int64_t task_count = elem_count / bias_len;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const T* p_input = input_data + task_idx * bias_len;
        T* p_output = output_data + task_idx * bias_len;
        T* p_tmp = tmp_data + task_idx * bias_len;

        for (int64_t index = 0; index < bias_len; index++) {
          T value = p_input[index] + bias_data[index];
          p_output[index] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
          p_tmp[index] = value * 0.5f;
        }

        MlasComputeTanh(p_output, p_output, bias_len);

        for (int64_t index = 0; index < bias_len; index++) {
          p_output[index] = p_tmp[index] * (p_output[index] + 1.0f);
        }
      },
      0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
