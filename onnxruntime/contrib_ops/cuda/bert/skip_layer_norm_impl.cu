/*
 The implementation of this file is based on skipLayerNorm plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Add SkipLayerNormKernelVec to
//                leverage vectorized load/write.
//                and templatize ComputeSkipLayerNorm for different
//                data types.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/layer_norm.cuh"
#include "contrib_ops/cuda/bert/skip_layer_norm_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
template <typename T>
T maybe2half(float x);

template <>
float maybe2half(float x) {
  return x;
}

template <>
half maybe2half(float x) {
  return __float2half_rn(x);
}

// Using only power of 2 numbers will lead to waste of compute for same size such as 768, which is a very common case
// in BERT. Ideally we can step by wrap_size * num_unroll, but listing too many steps will cause long compile time.
constexpr int kSizes[] = {32, 64, 128, 384, 768, 1024, 2048};
constexpr int kMinBlockSize = 32;
constexpr int kMaxBlockSize = 256;

int NextSize(int x) {
  size_t len = sizeof(kSizes) / sizeof(kSizes[0]);
  for (size_t i = 0; i < len; ++i) {
    if (x <= kSizes[i]) {
      return kSizes[i];
    }
  }
  return kSizes[len - 1];
}

template <typename T, int NumUnroll>
bool CanVectorized(T* output, T* sum_output, const T* input, const T* skip, const T* bias,
                   const T* gamma, const T* beta, const int ld, const int next_size) {
  constexpr int alignment = std::alignment_of<aligned_vector<T, NumUnroll>>::value;
  return ld % NumUnroll == 0 &&
         reinterpret_cast<uint64_t>(output) % alignment == 0 &&
         reinterpret_cast<uint64_t>(sum_output) % alignment == 0 &&
         reinterpret_cast<uint64_t>(input) % alignment == 0 &&
         reinterpret_cast<uint64_t>(skip) % alignment == 0 &&
         reinterpret_cast<uint64_t>(bias) % alignment == 0 &&
         reinterpret_cast<uint64_t>(gamma) % alignment == 0 &&
         reinterpret_cast<uint64_t>(beta) % alignment == 0 &&
         next_size / NumUnroll >= kMinBlockSize &&
         next_size / NumUnroll <= kMaxBlockSize;
}
}  // namespace

template <typename T, unsigned TPB, bool Simplified>
__global__ void SkipLayerNormKernel(
    T* output, T* sum_output, const T* input, const T* skip, const T* bias, const T* gamma, const T* beta, T epsilon,
    const int ld, int skip_size) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;
  const bool has_bias = (bias != nullptr);

  // Reduce sum of x and x^2, and the results are divided by ld.
  KeyValuePairSum pair_sum;
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;

    T val = input[idx];
    if (has_bias) {
      val += bias[i];
    }
    val += skip[idx % skip_size];

    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));

    if (sum_output != nullptr) {
      sum_output[idx] = val;
    }

    output[idx] = val;
  }

  if (Simplified) {
    SimplifiedLayerNorm<T, TPB>(thread_data.value, ld, offset, gamma, epsilon, output);
    return;
  }
  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

// Vectorized kernel
template <typename T, unsigned TPB, int ILP, bool Simplified>
__global__ void SkipLayerNormKernelSmall(
    T* output, T* sum_output, const T* input, const T* skip, const T* bias, const T* gamma, const T* beta, T epsilon,
    int ld, int skip_size) {
  const T rld = T(1.f / ld);
  const int idx = blockIdx.x * ld + threadIdx.x * ILP;

  using VecT = aligned_vector<T, ILP>;

  T skip_v[ILP], bias_v[ILP], sum_v[ILP];

  // load input to sum_v
  VecT* sum_val = reinterpret_cast<VecT*>(&sum_v);
  *sum_val = *reinterpret_cast<const VecT*>(&input[idx]);

  VecT* skip_val = reinterpret_cast<VecT*>(&skip_v);
  *skip_val = *reinterpret_cast<const VecT*>(&skip[idx % skip_size]);

  const bool has_bias = (bias != nullptr);
  if (has_bias) {
    VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
    *bias_val = *reinterpret_cast<const VecT*>(&bias[threadIdx.x * ILP]);
  }

  cub::KeyValuePair<T, T> thread_data(T(0.f), T(0.f));

  if (ILP * threadIdx.x < ld) {
    T rldval_sum = T(0.f);
    T rldvalsq_sum = T(0.f);
    const bool has_sum_output = (sum_output != nullptr);

#pragma unroll
    for (int i = 0; i < ILP; i++) {
      if (has_bias) {
        sum_v[i] += bias_v[i];
      }
      sum_v[i] += skip_v[i];

      const T rldval = rld * sum_v[i];
      rldval_sum += rldval;
      rldvalsq_sum += rldval * sum_v[i];
    }

    if (has_sum_output) {
      *(reinterpret_cast<VecT*>(&sum_output[idx])) = *reinterpret_cast<VecT*>(&sum_v);
    }

    thread_data = cub::KeyValuePair<T, T>(rldval_sum, rldvalsq_sum);
  }

  if (Simplified) {
    SimplifiedLayerNormSmall<T, TPB, ILP>(sum_v, thread_data.value, ld, idx, gamma, epsilon, output);
    return;
  }
  LayerNormSmall<T, TPB, ILP>(sum_v, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, bool Simplified>
void LaunchSkipLayerNormKernel(
    cudaStream_t stream, T* output, T* sum_output,
    const T* input, const T* skip, const T* bias, const T* gamma, const T* beta, float epsilon,
    int ld, int row_count, int skip_size) {
  const int next_size = NextSize(ld);
  const int grid_size = row_count;
  bool flag_vec2 = CanVectorized<T, 2>(output, sum_output, input, skip, bias, gamma, beta, ld, next_size);
  bool flag_vec4 = CanVectorized<T, 4>(output, sum_output, input, skip, bias, gamma, beta, ld, next_size);

#define LAUNCH_SKIP_LAYER_NORM_KERNEL_SMALL(num_unroll)                                                  \
  SkipLayerNormKernelSmall<T, block_size, num_unroll, Simplified><<<grid_size, block_size, 0, stream>>>( \
      output, sum_output, input, skip, bias, gamma, beta, maybe2half<T>(epsilon), ld, skip_size)

#define LAUNCH_SKIP_LAYER_NORM_KERNEL()                                                       \
  SkipLayerNormKernel<T, kMaxBlockSize, Simplified><<<grid_size, kMaxBlockSize, 0, stream>>>( \
      output, sum_output, input, skip, bias, gamma, beta, maybe2half<T>(epsilon), ld, skip_size)

#define CASE_NEXT_SIZE(next_size_value)               \
  case next_size_value: {                             \
    if (flag_vec4) {                                  \
      constexpr int block_size = next_size_value / 4; \
      LAUNCH_SKIP_LAYER_NORM_KERNEL_SMALL(4);         \
    } else if (flag_vec2) {                           \
      constexpr int block_size = next_size_value / 2; \
      LAUNCH_SKIP_LAYER_NORM_KERNEL_SMALL(2);         \
    } else {                                          \
      if (next_size_value <= kMaxBlockSize) {         \
        constexpr int block_size = next_size_value;   \
        LAUNCH_SKIP_LAYER_NORM_KERNEL_SMALL(1);       \
      } else {                                        \
        LAUNCH_SKIP_LAYER_NORM_KERNEL();              \
      }                                               \
    }                                                 \
  } break

  switch (next_size) {
    CASE_NEXT_SIZE(kSizes[0]);
    CASE_NEXT_SIZE(kSizes[1]);
    CASE_NEXT_SIZE(kSizes[2]);
    CASE_NEXT_SIZE(kSizes[3]);
    CASE_NEXT_SIZE(kSizes[4]);
    CASE_NEXT_SIZE(kSizes[5]);
    CASE_NEXT_SIZE(kSizes[6]);
  }

#undef CASE_NEXT_SIZE
#undef LAUNCH_SKIP_LAYER_NORM_KERNEL
#undef LAUNCH_SKIP_LAYER_NORM_KERNEL_SMALL
}

#define SKIPLAYERNORM_IMPL(T, Simplified)                                                                 \
  template void LaunchSkipLayerNormKernel<T, Simplified>(cudaStream_t stream, T * output, T * sum_output, \
                                                         const T* input, const T* skip, const T* bias,    \
                                                         const T* gamma, const T* beta, float epsilon,    \
                                                         int ld, int row_count, int skip_size);
SKIPLAYERNORM_IMPL(float, true);
SKIPLAYERNORM_IMPL(float, false);
SKIPLAYERNORM_IMPL(half, true);
SKIPLAYERNORM_IMPL(half, false);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
