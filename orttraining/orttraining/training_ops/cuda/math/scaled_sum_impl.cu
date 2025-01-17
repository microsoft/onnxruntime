// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/math/scaled_sum_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T, int NumUnroll, int InputCount, bool IsVectorized>
struct ScaledSumFunctor {
  ScaledSumFunctor(const std::vector<const T*>& inputs,
                   const std::vector<float>& scales,
                   int64_t N,
                   T* output) {
    output_data_ = output;
    N_ = static_cast<CUDA_LONG>(N);
    for (int i = 0; i < InputCount; i++) {
      inputs_[i] = inputs[i];
      scales_[i] = scales[i];
    }
  }

  __device__ __inline__ void operator()(CUDA_LONG idx) const {
    CUDA_LONG id = idx * NumUnroll;

    if (id >= N_) {
      return;
    }

    using LoadT = aligned_vector<T, NumUnroll>;
    T input_values[InputCount][NumUnroll];
    if (IsVectorized) {
      LoadT* input0_value_ptr = reinterpret_cast<LoadT*>(&input_values[0][0]);
      *input0_value_ptr = *reinterpret_cast<const LoadT*>(&inputs_[0][id]);

      LoadT* input1_value_ptr = reinterpret_cast<LoadT*>(&input_values[1][0]);
      *input1_value_ptr = *reinterpret_cast<const LoadT*>(&inputs_[1][id]);

      if (InputCount == 3) {
        LoadT* input2_value_ptr = reinterpret_cast<LoadT*>(&input_values[2][0]);
        *input2_value_ptr = *reinterpret_cast<const LoadT*>(&inputs_[2][id]);
      }

    } else {
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          input_values[0][i] = inputs_[0][li];
          input_values[1][i] = inputs_[1][li];
          if (InputCount == 3)
            input_values[2][i] = inputs_[2][li];
        }
      }
    }

    if (IsVectorized) {
      T output_value[NumUnroll];
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          if (InputCount == 3)
            output_value[i] = input_values[0][i] * static_cast<T>(scales_[0]) +
                              input_values[1][i] * static_cast<T>(scales_[1]) +
                              input_values[2][i] * static_cast<T>(scales_[2]);
          else
            output_value[i] = input_values[0][i] * static_cast<T>(scales_[0]) +
                              input_values[1][i] * static_cast<T>(scales_[1]);
        }
      }

      *reinterpret_cast<LoadT*>(&output_data_[id]) = *reinterpret_cast<LoadT*>(&output_value[0]);
    } else {
      T* output_value = output_data_ + id;
#pragma unroll
      for (int i = 0; i < NumUnroll; i++) {
        CUDA_LONG li = id + i;
        if (li < N_) {
          if (InputCount == 3)
            output_value[i] = input_values[0][i] * static_cast<T>(scales_[0]) +
                              input_values[1][i] * static_cast<T>(scales_[1]) +
                              input_values[2][i] * static_cast<T>(scales_[2]);

          else
            output_value[i] = input_values[0][i] * static_cast<T>(scales_[0]) +
                              input_values[1][i] * static_cast<T>(scales_[1]);
        }
      }
    }
  }

 private:
  const T* inputs_[InputCount];
  float scales_[InputCount];
  CUDA_LONG N_;
  T* output_data_;
};

template <typename FuncT>
__global__ void ScaledSumKernel(const FuncT functor) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  functor(idx);
}

template <typename T>
void ScaledSumImpl(cudaStream_t stream,
                   int64_t input_element_count,
                   const std::vector<const T*>& inputs,
                   const std::vector<float>& scales,
                   T* output_data) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(input_element_count, kBlockSize * kNumUnroll));
  constexpr int vec_alignment = std::alignment_of<aligned_vector<T, kNumUnroll>>::value;
  const bool use_vectorized = (input_element_count % kNumUnroll == 0) &&
                              (reinterpret_cast<uintptr_t>(output_data) % vec_alignment == 0) &&
                              (reinterpret_cast<uintptr_t>(inputs[0]) % vec_alignment == 0) &&
                              (reinterpret_cast<uintptr_t>(inputs[1]) % vec_alignment == 0) &&
                              (inputs.size() < 3 || (reinterpret_cast<uintptr_t>(inputs[2]) % vec_alignment == 0));

  const int input_count = static_cast<int>(inputs.size());
  using TwoInputTVectorizedFunctorType = ScaledSumFunctor<T, kNumUnroll, 2, true>;
  using TwoInputTNonVectorizedFunctorType = ScaledSumFunctor<T, kNumUnroll, 2, false>;
  using ThreeInputTVectorizedFunctorType = ScaledSumFunctor<T, kNumUnroll, 3, true>;
  using ThreeInputTNonVectorizedFunctorType = ScaledSumFunctor<T, kNumUnroll, 3, false>;

  if (input_count == 2) {
    if (use_vectorized) {
      ScaledSumKernel<TwoInputTVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          TwoInputTVectorizedFunctorType(inputs, scales, input_element_count, output_data));
    } else {
      ScaledSumKernel<TwoInputTNonVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          TwoInputTNonVectorizedFunctorType(inputs, scales, input_element_count, output_data));
    }
  } else if (input_count == 3) {
    if (use_vectorized) {
      ScaledSumKernel<ThreeInputTVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          ThreeInputTVectorizedFunctorType(inputs, scales, input_element_count, output_data));
    } else {
      ScaledSumKernel<ThreeInputTNonVectorizedFunctorType><<<blocksPerGrid, kBlockSize, 0, stream>>>(
          ThreeInputTNonVectorizedFunctorType(inputs, scales, input_element_count, output_data));
    }

  } else {
    ORT_THROW("Unsupported input count: ", input_count);
  }
}

#define SPECIALIZE_SCALED_SUM_IMPL(T)                                 \
  template void ScaledSumImpl<T>(cudaStream_t stream,                 \
                                 int64_t input_element_count,         \
                                 const std::vector<const T*>& inputs, \
                                 const std::vector<float>& scales,    \
                                 T* output_data);

SPECIALIZE_SCALED_SUM_IMPL(half);
SPECIALIZE_SCALED_SUM_IMPL(float);
SPECIALIZE_SCALED_SUM_IMPL(double);
SPECIALIZE_SCALED_SUM_IMPL(BFloat16);

#undef SPECIALIZE_SCALED_SUM_IMPL

}  // namespace cuda
}  // namespace onnxruntime
