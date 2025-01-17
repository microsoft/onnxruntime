// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/math/batch_scale_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T, int NumUnroll, int OutputCount, bool IsVectorized>
struct BatchScaleFunctor {
  BatchScaleFunctor(const T* input,
                    const std::vector<float>& scales,
                    int64_t N,
                    const std::vector<T*>& outputs)
      : N_(static_cast<CUDA_LONG>(N)),
        input_data_(input) {
    for (int i = 0; i < OutputCount; i++) {
      outputs_[i] = outputs[i];
      scales_[i] = scales[i];
    }
  }

  __device__ __inline__ void operator()(CUDA_LONG idx) const {
    CUDA_LONG id = idx * NumUnroll;

    if (id >= N_) {
      return;
    }

    using LoadT = aligned_vector<T, NumUnroll>;

    T input0_value[NumUnroll];
    if (IsVectorized) {
      LoadT* input0_value_ptr = reinterpret_cast<LoadT*>(&input0_value[0]);
      *input0_value_ptr = *reinterpret_cast<const LoadT*>(&input_data_[id]);
    } else {
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          input0_value[i] = input_data_[li];
        }
      }
    }

    if (IsVectorized) {
      T output_values[OutputCount][NumUnroll];
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          output_values[0][i] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[0]);
          output_values[1][i] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[1]);
          if (OutputCount == 3)
            output_values[2][i] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[2]);
        }
      }
      *reinterpret_cast<LoadT*>(&outputs_[0][id]) = *reinterpret_cast<LoadT*>(&output_values[0][0]);
      *reinterpret_cast<LoadT*>(&outputs_[1][id]) = *reinterpret_cast<LoadT*>(&output_values[1][0]);
      if (OutputCount == 3)
        *reinterpret_cast<LoadT*>(&outputs_[2][id]) = *reinterpret_cast<LoadT*>(&output_values[2][0]);

    } else {
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          outputs_[0][li] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[0]);
          outputs_[1][li] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[1]);
          if (OutputCount == 3)
            outputs_[2][li] = static_cast<T>(static_cast<float>(input0_value[i]) * scales_[2]);
        }
      }
    }
  }

 private:
  T* outputs_[OutputCount];
  float scales_[OutputCount];
  const CUDA_LONG N_;
  const T* input_data_;
};

template <typename FuncT>
__global__ void BatchScaleKernel(const FuncT functor) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  functor(idx);
}

template <typename T>
void BatchScaleImpl(cudaStream_t stream,
                    int64_t input_element_count,
                    const T* input_data,
                    const std::vector<float>& scales,
                    const std::vector<T*>& outputs) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(input_element_count, kBlockSize * kNumUnroll));
  constexpr int vec_alignment = std::alignment_of<aligned_vector<T, kNumUnroll>>::value;
  const bool use_vectorized = (input_element_count % kNumUnroll == 0) &&
                              (reinterpret_cast<uintptr_t>(input_data) % vec_alignment == 0) &&
                              (reinterpret_cast<uintptr_t>(outputs[0]) % vec_alignment == 0) &&
                              (reinterpret_cast<uintptr_t>(outputs[1]) % vec_alignment == 0) &&
                              (outputs.size() < 3 || (reinterpret_cast<uintptr_t>(outputs[2]) % vec_alignment == 0));

  const int output_count = static_cast<int>(outputs.size());
  using TwoOutputVectorizedFunctorType = BatchScaleFunctor<T, kNumUnroll, 2, true>;
  using TwoOutputNonVectorizedFunctorType = BatchScaleFunctor<T, kNumUnroll, 2, false>;
  using ThreeOutputVectorizedFunctorType = BatchScaleFunctor<T, kNumUnroll, 3, true>;
  using ThreeOutputNonVectorizedFunctorType = BatchScaleFunctor<T, kNumUnroll, 3, false>;

  if (output_count == 2) {
    if (use_vectorized)
      BatchScaleKernel<TwoOutputVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          TwoOutputVectorizedFunctorType(input_data, scales, input_element_count, outputs));
    else
      BatchScaleKernel<TwoOutputNonVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          TwoOutputNonVectorizedFunctorType(input_data, scales, input_element_count, outputs));
  } else if (output_count == 3) {
    if (use_vectorized) {
      BatchScaleKernel<ThreeOutputVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          ThreeOutputVectorizedFunctorType(input_data, scales, input_element_count, outputs));
    } else {
      BatchScaleKernel<ThreeOutputNonVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          ThreeOutputNonVectorizedFunctorType(input_data, scales, input_element_count, outputs));
    }

  } else {
    ORT_THROW("Unsupported output count: ", output_count);
  }
}

#define SPECIALIZE_BATCH_SCALE_IMPL(T)                              \
  template void BatchScaleImpl<T>(cudaStream_t stream,              \
                                  int64_t input_element_count,      \
                                  const T* input_data,              \
                                  const std::vector<float>& scales, \
                                  const std::vector<T*>& outputs);

SPECIALIZE_BATCH_SCALE_IMPL(half);
SPECIALIZE_BATCH_SCALE_IMPL(float);
SPECIALIZE_BATCH_SCALE_IMPL(double);
SPECIALIZE_BATCH_SCALE_IMPL(BFloat16);

#undef SPECIALIZE_BATCH_SCALE_IMPL

}  // namespace cuda
}  // namespace onnxruntime
