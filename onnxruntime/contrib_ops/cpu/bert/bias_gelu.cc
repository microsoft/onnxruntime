// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_gelu.h"
#include "bias_gelu_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    BiasGelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasGelu<float, false>);

// FastGelu uses approximation for Gelu. The formula is 0.5 * (1 + Tanh(x * (C * x * x + B))) * x.
static constexpr float B = 0.7978845608028654f;    // sqrt(2.0 / M_PI)
static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0 / M_PI)

template <typename T, bool use_approximation>
Status BiasGelu<T, use_approximation>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->template Data<T>();
  int64_t elem_count = input->Shape().Size();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->template MutableData<T>();

  const Tensor* bias = context->Input<Tensor>(1);
  if (nullptr == bias) {
    // FastGelu allows optional bias. Here we split input data into chunks. Each chunk
    // has N elements (except the last chunk), and use thread pool to parallel chunks.
    // N = 4096 is selected based on performance test results on input shape 1x128x768.
    ORT_ENFORCE(use_approximation);
    if (use_approximation) {
      static const int64_t length_per_task = 4096;
      int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
      concurrency::ThreadPool::TryBatchParallelFor(
          context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
          [&](ptrdiff_t task_idx) {
            const auto start = task_idx * length_per_task;
            const T* p_input = input_data + start;
            T* p_output = output_data + start;
            int64_t count = std::min(length_per_task, elem_count - start);

            for (int64_t i = 0; i < count; i++) {
              T value = p_input[i];
              p_output[i] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
            }

            MlasComputeTanh(p_output, p_output, count);

            for (int64_t i = 0; i < count; i++) {
              p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
            }
          },
          0);
    }
    return Status::OK();
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

        AddBiasGelu(p_input, bias_data, p_tmp, p_output, bias_len);
      },
      0);

  return Status::OK();
}

template <typename T, bool use_approximation>
void BiasGelu<T, use_approximation>::AddBiasGelu(
    const T* input, const T* bias, T* temp, T* output, int64_t count) const {
  if (use_approximation) {
    for (int64_t i = 0; i < count; i++) {
      T value = input[i] + bias[i];
      output[i] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
      temp[i] = value * 0.5f;
    }

    MlasComputeTanh(output, output, count);

    for (int64_t i = 0; i < count; i++) {
      output[i] = temp[i] * (output[i] + 1.0f);
    }
  } else {  // BiasGelu
    for (int64_t i = 0; i < count; i++) {
      T value = input[i] + bias[i];
      output[i] = value * static_cast<T>(M_SQRT1_2);
      temp[i] = value * 0.5f;
    }

    MlasComputeErf(output, output, count);

    for (int64_t i = 0; i < count; i++) {
      output[i] = temp[i] * (output[i] + 1.0f);
    }
  }
}

// Instantiation for BiasGelu
template class BiasGelu<float, false>;

// Instantiation for FastGelu
template class BiasGelu<float, true>;

}  // namespace contrib
}  // namespace onnxruntime
