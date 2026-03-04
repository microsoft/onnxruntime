// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/cpuid_info.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#if defined(MLAS_TARGET_AMD64) && defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/tensor/gelu.h"

using onnxruntime::narrow;
using namespace onnxruntime::common;

namespace onnxruntime {

namespace {

template <typename T>
inline void GeluPreProcessFusedTanh(const T* input, T* transformed, size_t count, T b, T c) {
  for (size_t i = 0; i < count; ++i) {
    const T x = input[i];
    transformed[i] = x * (c * x * x + b);
  }
}

template <typename T>
inline void GeluPreProcessFusedErf(const T* input, T* transformed, size_t count, T scale) {
  for (size_t i = 0; i < count; ++i) {
    transformed[i] = input[i] * scale;
  }
}

template <>
inline void GeluPreProcessFusedTanh<float>(const float* input, float* transformed, size_t count, float b, float c) {
#if defined(MLAS_TARGET_AMD64) && defined(__AVX512F__)
  static const bool has_avx512f = CPUIDInfo::GetCPUIDInfo().HasAVX512f();
  if (has_avx512f) {
    constexpr size_t kSimdWidth = 16;
    const __m512 bv = _mm512_set1_ps(b);
    const __m512 cv = _mm512_set1_ps(c);

    size_t i = 0;
    for (; i + kSimdWidth <= count; i += kSimdWidth) {
      const __m512 x = _mm512_loadu_ps(input + i);
      const __m512 x2 = _mm512_mul_ps(x, x);
      __m512 y = _mm512_fmadd_ps(cv, x2, bv);
      y = _mm512_mul_ps(y, x);
      _mm512_storeu_ps(transformed + i, y);
    }

    for (; i < count; ++i) {
      const float x = input[i];
      transformed[i] = x * (c * x * x + b);
    }
    return;
  }
#endif

  for (size_t i = 0; i < count; ++i) {
    const float x = input[i];
    transformed[i] = x * (c * x * x + b);
  }
}

template <>
inline void GeluPreProcessFusedErf<float>(const float* input, float* transformed, size_t count, float scale) {
#if defined(MLAS_TARGET_AMD64) && defined(__AVX512F__)
  static const bool has_avx512f = CPUIDInfo::GetCPUIDInfo().HasAVX512f();
  if (has_avx512f) {
    constexpr size_t kSimdWidth = 16;
    const __m512 sv = _mm512_set1_ps(scale);

    size_t i = 0;
    for (; i + kSimdWidth <= count; i += kSimdWidth) {
      const __m512 x = _mm512_loadu_ps(input + i);
      const __m512 y = _mm512_mul_ps(x, sv);
      _mm512_storeu_ps(transformed + i, y);
    }

    for (; i < count; ++i) {
      transformed[i] = input[i] * scale;
    }
    return;
  }
#endif

  for (size_t i = 0; i < count; ++i) {
    transformed[i] = input[i] * scale;
  }
}

template <typename T>
inline void GeluPostProcessFused(const T* input, T* transformed, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    transformed[i] = static_cast<T>(0.5f) * input[i] * (transformed[i] + static_cast<T>(1.0f));
  }
}

template <>
inline void GeluPostProcessFused<float>(const float* input, float* transformed, size_t count) {
#if defined(MLAS_TARGET_AMD64) && defined(__AVX512F__)
  static const bool has_avx512f = CPUIDInfo::GetCPUIDInfo().HasAVX512f();
  if (has_avx512f) {
    constexpr size_t kSimdWidth = 16;
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);

    size_t i = 0;
    for (; i + kSimdWidth <= count; i += kSimdWidth) {
      const __m512 x = _mm512_loadu_ps(input + i);
      __m512 y = _mm512_loadu_ps(transformed + i);
      y = _mm512_add_ps(y, one);
      y = _mm512_mul_ps(y, x);
      y = _mm512_mul_ps(y, half);
      _mm512_storeu_ps(transformed + i, y);
    }

    for (; i < count; ++i) {
      transformed[i] = 0.5f * input[i] * (transformed[i] + 1.0f);
    }

    return;
  }
#endif

  for (size_t i = 0; i < count; ++i) {
    transformed[i] = 0.5f * input[i] * (transformed[i] + 1.0f);
  }
}

}  // namespace

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

          GeluPreProcessFusedTanh(p_input, p_output, narrow<size_t>(count),
                                  static_cast<T>(B), static_cast<T>(C));
          MlasComputeTanh(p_output, p_output, narrow<size_t>(count));
          GeluPostProcessFused(p_input, p_output, narrow<size_t>(count));
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

          GeluPreProcessFusedErf(p_input, p_output, narrow<size_t>(count),
                                 static_cast<T>(M_SQRT1_2));
          MlasComputeErf(p_output, p_output, narrow<size_t>(count));
          GeluPostProcessFused(p_input, p_output, narrow<size_t>(count));
        },
        0);
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}

}  // namespace onnxruntime
