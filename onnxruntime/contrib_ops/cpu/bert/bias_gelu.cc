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
#include <cmath>
#include <numbers>
using onnxruntime::narrow;
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    BiasGelu,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasGelu<float, false>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    BiasGelu,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    BiasGelu<MLFloat16, false>);

// FastGelu uses approximation for Gelu. The formula is 0.5 * (1 + Tanh(x * (C * x * x + B))) * x.
static constexpr float B = 0.7978845608028654f;    // sqrt(2.0 / M_PI)
static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0 / M_PI)

template <typename T, bool use_approximation>
Status BiasGelu<T, use_approximation>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->Data<T>();
  int64_t elem_count = input->Shape().Size();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  // An empty input (and, correspondingly, an empty bias) is a legal degenerate case per the
  // ONNX shape model; treat it as a no-op rather than let elem_count / bias_len divide by zero below.
  if (elem_count == 0) {
    return Status::OK();
  }

  const Tensor* bias = context->Input<Tensor>(1);
  if (nullptr == bias) {
    // FastGelu allows optional bias. Here we split input data into chunks. Each chunk
    // has N elements (except the last chunk), and use thread pool to parallel chunks.
    // N = 4096 is selected based on performance test results on input shape 1x128x768.
    ORT_ENFORCE(use_approximation);
    if (use_approximation) {
      static constexpr int64_t length_per_task = 4096;
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

            MlasComputeTanh(p_output, p_output, narrow<size_t>(count));

            for (int64_t i = 0; i < count; i++) {
              p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
            }
          },
          0);
    }
    return Status::OK();
  }

  const T* bias_data = bias->Data<T>();
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

    MlasComputeTanh(output, output, narrow<size_t>(count));

    for (int64_t i = 0; i < count; i++) {
      output[i] = temp[i] * (output[i] + 1.0f);
    }
  } else {  // BiasGelu
    for (int64_t i = 0; i < count; i++) {
      T value = input[i] + bias[i];
      output[i] = value * (T{1} / std::numbers::sqrt2_v<T>);
      temp[i] = value * 0.5f;
    }

    MlasComputeErf(output, output, narrow<size_t>(count));

    for (int64_t i = 0; i < count; i++) {
      output[i] = temp[i] * (output[i] + 1.0f);
    }
  }
}

// temp holds 2 * count elements: input + bias, then scratch space for MlasComputeFP16Gelu.
static void AddBiasGeluFp16(const MLFloat16* input, const MLFloat16* bias, MLFloat16* temp, MLFloat16* output,
                            int64_t count, MLAS_GELU_ALGORITHM algo) {
  const size_t n = narrow<size_t>(count);
  MLFloat16* sum = temp;
  MLFloat16* scratch = temp + count;
  if (MlasFp16AccelerationSupported()) {
    MlasEltwiseAdd<MLAS_FP16>(input, bias, sum, n);
  } else {
    for (size_t i = 0; i < n; i++) {
      sum[i] = MLFloat16(input[i].ToFloat() + bias[i].ToFloat());
    }
  }
  MlasComputeFP16Gelu(sum, output, scratch, n, algo);
}

// Has to be specialized before Compute, the generic version doesn't build for MLFloat16.
template <>
void BiasGelu<MLFloat16, false>::AddBiasGelu(
    const MLFloat16* input, const MLFloat16* bias, MLFloat16* temp, MLFloat16* output, int64_t count) const {
  AddBiasGeluFp16(input, bias, temp, output, count, MlasGeluErf);
}

// The generic Compute has a no-bias FastGelu path that doesn't compile for MLFloat16.
template <>
Status BiasGelu<MLFloat16, false>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  int64_t elem_count = input->Shape().Size();

  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  const Tensor* bias = context->Input<Tensor>(1);
  ORT_ENFORCE(bias != nullptr, "BiasGelu requires a bias input for the non-approximation variant");

  const MLFloat16* bias_data = bias->Data<MLFloat16>();
  int64_t bias_len = bias->Shape().Size();

  if (elem_count == 0) {
    return Status::OK();
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr buffer = BufferUniquePtr(
      alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * elem_count * 2),
      BufferDeleter(alloc));
  MLFloat16* tmp_data = static_cast<MLFloat16*>(buffer.get());

  int64_t task_count = elem_count / bias_len;
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const MLFloat16* p_input = input_data + task_idx * bias_len;
        MLFloat16* p_output = output_data + task_idx * bias_len;
        MLFloat16* p_tmp = tmp_data + task_idx * bias_len * 2;
        AddBiasGelu(p_input, bias_data, p_tmp, p_output, bias_len);
      },
      0);

  return Status::OK();
}

template <>
Status BiasGelu<MLFloat16, true>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  int64_t elem_count = input->Shape().Size();

  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  if (elem_count == 0) {
    return Status::OK();
  }

  const Tensor* bias = context->Input<Tensor>(1);
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr buffer = BufferUniquePtr(
      alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * elem_count * 2),
      BufferDeleter(alloc));
  MLFloat16* tmp_data = static_cast<MLFloat16*>(buffer.get());

  if (nullptr == bias) {
    static constexpr int64_t length_per_task = 4096;
    int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          int64_t count = std::min(length_per_task, elem_count - start);
          MlasComputeFP16Gelu(input_data + start, output_data + start, tmp_data + start,
                              narrow<size_t>(count), MlasGeluTanh);
        },
        0);
    return Status::OK();
  }

  const MLFloat16* bias_data = bias->Data<MLFloat16>();
  int64_t bias_len = bias->Shape().Size();
  int64_t task_count = elem_count / bias_len;
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        AddBiasGeluFp16(input_data + task_idx * bias_len, bias_data, tmp_data + task_idx * bias_len * 2,
                        output_data + task_idx * bias_len, bias_len, MlasGeluTanh);
      },
      0);
  return Status::OK();
}

// Registered here so the Compute specialization above is visible.
ONNX_OPERATOR_TYPED_KERNEL_EX(
    FastGelu,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    BiasGelu<MLFloat16, true>);

// Instantiation for BiasGelu
template class BiasGelu<float, false>;

// Instantiation for FastGelu
template class BiasGelu<float, true>;

}  // namespace contrib
}  // namespace onnxruntime
