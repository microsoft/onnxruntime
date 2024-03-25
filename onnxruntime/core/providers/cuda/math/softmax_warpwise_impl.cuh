/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The code below is mostly copied from Pytorch PersistentSoftmax.cuh

#pragma once
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "GPU_WARP_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architecures, but there is no guarantee this will not change for future arch.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_log_softmax) {
        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
      } else {
        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    if (is_log_softmax) sum[i] = max_value[i] + std::log((float)(sum[i]));
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

// softmax_warp_forward uses register to store data in fp32 even when data is fp16, which will cause register resource oversubscription when data is large,
// and will lead to low CUDA warp occupancy and thus a poor kernel performance.
// softmax_warp_forward_resource_efficient is implemented to solve the issue, it caches data in original data type, and casts it into fp32 when needed,
// the idea is like we use recomputation to save resource usage.
template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward_resource_efficient(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // 1 cuda block only processes one row and contains 1 cuda warp only.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;

  int local_idx = threadIdx.x;
  src += blockIdx.x * stride + local_idx;
  dst += blockIdx.x * stride + local_idx;
  extern __shared__ unsigned char smem[];
  input_t(&elements)[WARP_ITERATIONS][WARP_SIZE] = *reinterpret_cast<input_t(*)[WARP_ITERATIONS][WARP_SIZE]>(smem);
#pragma unroll
  for (int it = 0; it < WARP_ITERATIONS; ++it) {
    int element_index = local_idx + it * WARP_SIZE;
    if (element_index < element_count) {
      elements[it][local_idx] = src[it * WARP_SIZE];
    } else {
      static_assert(!std::is_same<acc_t, half>::value, "acc_t can no be half, as the infinity function will return 0 instead of inf");
      elements[it][local_idx] = (input_t)-std::numeric_limits<acc_t>::infinity();
    }
  }
  // compute max_value
  input_t max_value = elements[0][local_idx];
#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    max_value = (max_value > elements[it][local_idx]) ? max_value : elements[it][local_idx];
  }
  warp_reduce<input_t, 1, WARP_SIZE, Max>(&max_value);
  // compute sum
  acc_t sum{0.0f};
  // #pragma unroll
  // "exp" contains many instructions, if we unroll the loop then cuda warp will be stalled because of icache miss and thus lower the perf.
  for (int it = 0; it < WARP_ITERATIONS; ++it) {
    int element_index = local_idx + it * WARP_SIZE;
    if (element_index >= element_count)
      break;
    if (is_log_softmax) {
      sum += std::exp((float)(elements[it][local_idx] - max_value));
    } else {
      acc_t tmp = std::exp((float)(elements[it][local_idx] - max_value));
      elements[it][local_idx] = tmp;
      sum += tmp;
    }
  }
  warp_reduce<acc_t, 1, WARP_SIZE, Add>(&sum);
  // store result
  if (is_log_softmax) sum = static_cast<acc_t>(max_value) + std::log((float)(sum));
  // do the reciprocal once, so the div operation can be replaced by mul.
  acc_t invsum = static_cast<acc_t>(1.0f / sum);
#pragma unroll
  for (int it = 0; it < WARP_ITERATIONS; ++it) {
    int element_index = local_idx + it * WARP_SIZE;
    if (element_index < element_count) {
      if (is_log_softmax) {
        dst[it * WARP_SIZE] = (float)elements[it][local_idx] - sum;
      } else {
        dst[it * WARP_SIZE] = (float)elements[it][local_idx] * invsum;
      }
    } else {
      break;
    }
  }
}

}  // namespace cuda
}  // namespace onnxruntime
