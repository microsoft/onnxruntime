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
#include "core/mlas/lib/fp16_common.h"
#if defined(MLAS_NEON_INTRINSICS)
#include "core/mlas/lib/erf_neon_fp16.h"
#endif

#ifdef MLAS_USE_SVE
#include "core/mlas/lib/sve/mlasi_sve.h"
#endif

using onnxruntime::narrow;
using namespace onnxruntime::common;

namespace onnxruntime {

// May revisit the implementations to support inplace computation, if needed.

#define ADD_TYPED_GELU_OP(data_type)                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                         \
      Gelu,                                                               \
      20,                                                                 \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Gelu<data_type>)

ADD_TYPED_GELU_OP(float);
ADD_TYPED_GELU_OP(MLFloat16);

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

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

void ComputeGeluFp16_NEON(const MLFloat16* input, MLFloat16* output, MLFloat16* temp, int64_t count, const std::string& algo) {
  const float16_t v_half1 = 0.5f;
  const float16_t v_one1 = 1.0f;
  const float16_t v_sqrt1_21 = static_cast<float>(M_SQRT1_2);
  const float16_t v_B1 = 0.7978845608028654f;
  const float16_t v_C1 = 0.035677408136300125f;
  const float16_t c1 = 5.0f;
  const float16_t c2 = -5.0f;
  const MLAS_FLOAT16X8 v_half = MlasBroadcastF16Float16x8(v_half1);
  const MLAS_FLOAT16X8 v_one = MlasBroadcastF16Float16x8(v_one1);
  const MLAS_FLOAT16X8 v_sqrt1_2 = MlasBroadcastF16Float16x8(v_sqrt1_21);
  const MLAS_FLOAT16X8 v_B = MlasBroadcastF16Float16x8(v_B1);
  const MLAS_FLOAT16X8 v_C = MlasBroadcastF16Float16x8(v_C1);

  int64_t i = 0;

  if (algo == "tanh") {
    // Preprocess input into temp[] for tanh
    for (; i + 7 < count; i += 8) {
      MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
      MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(x, x);
      MLAS_FLOAT16X8 inner = MlasMultiplyAddFloat16(v_C, x2, v_B);  // B + C * x^2
      MLAS_FLOAT16X8 tanh_arg = MlasMultiplyFloat16(x, inner);      // x * (B + C * x^2)
      tanh_arg = MlasMaximumFloat16(MlasBroadcastF16Float16x8(c2), MlasMinimumFloat16(tanh_arg, MlasBroadcastF16Float16x8(c1)));
      MlasStoref16Float16x8(reinterpret_cast<float16_t*>(temp + i), tanh_arg);
    }

    // Tail
    for (; i < count; ++i) {
      float x = static_cast<float>(input[i]);
      float inner = x * (0.7979f + 0.03568f * x * x);
      inner = std::max(-5.0f, std::min(5.0f, inner));
      temp[i] = static_cast<MLFloat16>(inner);
    }

    // Tanh processing
    MlasComputeTanh<MLAS_FP16>(reinterpret_cast<const MLAS_FP16*>(temp),
                               reinterpret_cast<MLAS_FP16*>(temp),
                               count);

  } else if (algo == "none") {
    // Preprocess input into temp[] for erf
    for (i = 0; i + 7 < count; i += 8) {
      MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
      MLAS_FLOAT16X8 scaled = MlasMultiplyFloat16(x, v_sqrt1_2);
      MlasStoref16Float16x8(reinterpret_cast<float16_t*>(temp + i), scaled);
    }

    // Tail
    for (; i < count; ++i) {
      float x = static_cast<float>(input[i]);
      temp[i] = static_cast<MLFloat16>(x * 0.70710678f);
    }

    // Erf processing
    MlasNeonErfKernelFp16(reinterpret_cast<const _mlas_fp16_*>(temp),
                          reinterpret_cast<_mlas_fp16_*>(temp),
                          count);
  }

  // Final GELU output = 0.5 * x * (1 + tanh|erf)
  i = 0;
  for (; i + 7 < count; i += 8) {
    MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
    MLAS_FLOAT16X8 t = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(temp + i));
    MLAS_FLOAT16X8 result = MlasMultiplyFloat16(v_half, MlasMultiplyFloat16(x, MlasAddFloat16(v_one, t)));
    MlasStoref16Float16x8(reinterpret_cast<float16_t*>(output + i), result);
  }

  for (; i < count; ++i) {
    float x = static_cast<float>(input[i]);
    float t = static_cast<float>(temp[i]);
    float gelu = 0.5f * x * (1.0f + t);
    output[i] = static_cast<MLFloat16>(gelu);
  }
}

#endif

template <>
Status Gelu<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  if (approximation_algorithm_ != "tanh" && approximation_algorithm_ != "none") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
  }

  // Alignment and buffer size for aligned_alloc
  constexpr size_t alignment = 64;
  size_t buffer_size = elem_count * sizeof(MLFloat16);
  size_t aligned_size = ((buffer_size + alignment - 1) / alignment) * alignment;
  auto deleter = [](MLFloat16* p) { std::free(p); };
  std::unique_ptr<MLFloat16, decltype(deleter)> temp_fp16_aligned(
      reinterpret_cast<MLFloat16*>(std::aligned_alloc(alignment, aligned_size)),
      deleter);
  if (temp_fp16_aligned == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate aligned temporary buffer.");
  }

  concurrency::ThreadPool::TryBatchParallelFor(
      tp,
      static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const MLFloat16* p_input = input_data + start;
        MLFloat16* p_output = output_data + start;
        int64_t count = std::min(length_per_task, elem_count - start);

#if defined(MLAS_USE_SVE) || defined(MLAS_NEON_INTRINSICS)
        MLFloat16* p_temp = temp_fp16_aligned.get() + start;
        bool done = false;

#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
          ComputeGeluFp16_SVE(p_input, p_output, p_temp, count, approximation_algorithm_);
          done = true;
        }
#endif

#if defined(MLAS_NEON_INTRINSICS)
        if (!done) {
          ComputeGeluFp16_NEON(p_input, p_output, p_temp, count, approximation_algorithm_);
          done = true;
        }
#endif
#else
        for (int64_t i = 0; i < count; ++i) {
          float x = static_cast<float>(p_input[i]);
          float gelu_val;
          if (approximation_algorithm_ == "tanh") {
            // GELU approx with tanh
            const float B = 0.7978845608f;
            const float C = 0.044715f * B;
            float tanh_arg = x * (B + C * x * x);
            float tanh_res = std::tanh(tanh_arg);
            gelu_val = 0.5f * x * (1 + tanh_res);
          } else {  // "none"
            gelu_val = 0.5f * x * (1 + std::erf(x * static_cast<float>(M_SQRT1_2)));
          }
          p_output[i] = MLFloat16(gelu_val);
        }
#endif
      },
      0);
  return Status::OK();
}

}  // namespace onnxruntime
