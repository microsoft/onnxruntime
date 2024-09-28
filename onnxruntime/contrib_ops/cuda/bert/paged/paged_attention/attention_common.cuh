// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/tensor.hpp"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/mutex.cuh"
#include "contrib_ops/cuda/bert/paged/type_convert.cuh"
#include "contrib_ops/cuda/bert/paged/warp_utilities.cuh"

namespace onnxruntime::contrib::paged {

// Utility function for attention softmax.
template <int NUM_WARPS>
__forceinline__ __device__ float
block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / constant::WarpSize;
  int lane = threadIdx.x % constant::WarpSize;

  CUTE_UNROLL
  for (int mask = constant::WarpSize / 2; mask >= 1; mask /= 2) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  if (lane == 0) {
    red_smem[warp] = sum;
  }

  __syncthreads();

  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  CUTE_UNROLL
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return SHFL_SYNC(sum, 0);
}

template <int NumThreads, int GroupSize, int NumWarps, int NumLogits>
__forceinline__ __device__ void
softmax_cta(
    float* reduction_buffer, float qk_max_of_group,
    float* logits,
    int start_tok_idx, int end_tok_idx,  // valid range
    float2* max_sum = nullptr            // optional output for max and sum of the chunk
) {
  auto lane_idx = lane_id();
  float qk_max = qk_max_of_group;
  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The leading thread of each group already has its max qk value.
  CUTE_UNROLL
  for (int mask = constant::WarpSize / 2; mask >= GroupSize; mask /= 2) {
    qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane_idx == 0) {
    reduction_buffer[warp_id()] = qk_max;
  }
  __syncthreads();

  // Get the max qk value for the sequence.
  qk_max = lane_idx < NumWarps ? reduction_buffer[lane_idx] : std::numeric_limits<float>::lowest();
  CUTE_UNROLL
  for (int mask = NumWarps / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  CUTE_UNROLL
  for (int i = threadIdx.x; i < NumLogits; i += NumThreads) {
    float val = (start_tok_idx <= i && i < end_tok_idx) ? __expf(logits[i] - qk_max) : 0.0f;
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NumWarps>(&reduction_buffer[NumWarps], exp_sum);

  // Compute softmax.
  const float inv_sum = __frcp_rn(exp_sum + 1e-6f);
  CUTE_UNROLL
  for (int i = threadIdx.x; i < NumLogits; i += NumThreads) {
    logits[i] *= inv_sum;
  }

  if (max_sum != nullptr && threadIdx.x == 0) {
    *max_sum = float2{qk_max, exp_sum};
  }
  __syncthreads();
}

#if 0
template <int NumLogits>
__forceinline__ __device__ void
softmax_warp(
    float* logits,
    int start_tok_idx, int end_tok_idx,  // valid range
    float2* max_sum = nullptr            // optional output for max and sum of the chunk
) {
  auto lane_idx = lane_id();

  // compute max along logits
  float max_val = std::numeric_limits<float>::lowest();
  CUTE_UNROLL
  for (int i = lane_idx; i < NumLogits; i += constant::WarpSize) {
    float val = (start_tok_idx <= i && i < end_tok_idx) ? logits[i] : std::numeric_limits<float>::lowest();
    max_val = fmaxf(max_val, val);
  }
  max_val = warp::reduce<constant::WarpSize>(max_val, [](float a, float b) { return fmaxf(a, b); });
  max_val = warp::broadcast<constant::WarpSize>(max_val);

  // compute the sum of exp
  float exp_sum = 0.f;
  CUTE_UNROLL
  for (int i = lane_idx; i < NumLogits; i += constant::WarpSize) {
    float val = (start_tok_idx <= i && i < end_tok_idx) ? __expf(logits[i] - max_val) : 0.0f;
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = warp::reduce<constant::WarpSize>(exp_sum, [](float a, float b) { return a + b; });
  exp_sum = warp::broadcast<constant::WarpSize>(exp_sum);

  // apply softmax denominator
  const float inv_sum = __frcp_rn(exp_sum + 1e-6f);
  CUTE_UNROLL
  for (int i = lane_idx; i < NumLogits; i += constant::WarpSize) {
    logits[i] *= inv_sum;
  }

  if (max_sum != nullptr && lane_idx == 0) {
    *max_sum = float2{max_val, exp_sum};
  }
}
#else
template <int NumLogits>
__forceinline__ __device__ void
softmax_warp(
    float* logits_ptr,
    int start_tok_idx, int end_tok_idx,  // valid range
    float2* max_sum = nullptr            // optional output for max and sum of the chunk
) {
  static_assert(ceil_div(NumLogits, constant::WarpSize) <= 96);

  auto lane_idx = lane_id();

  constexpr auto LogitsLayout = make_layout(Int<NumLogits>{});
  auto tiled_copy = make_tiled_copy(
      Copy_Atom<UniversalCopy<float>, float>{},
      make_layout(Int<constant::WarpSize>{}),
      make_layout(_1{})
  );
  auto thr_copy = tiled_copy.get_thread_slice(lane_idx);
  auto logits = thr_copy.partition_D(make_tensor(make_smem_ptr(logits_ptr), LogitsLayout));
  auto tcoord = thr_copy.partition_D(make_identity_tensor(shape(LogitsLayout)));

  auto regs = make_tensor_like(logits);
  fill(regs, std::numeric_limits<float>::lowest());
  CUTE_UNROLL
  for (int i = 0; i < size(regs); i++) {
    if (start_tok_idx <= tcoord(i) && tcoord(i) < end_tok_idx) {
      regs(i) = logits(i);
    }
  }

  float max_val = std::numeric_limits<float>::lowest();
  CUTE_UNROLL
  for (int i = 0; i < size(regs); i++) {
    max_val = fmaxf(max_val, regs(i));
  }
  max_val = warp::reduce<constant::WarpSize>(max_val, [](float a, float b) { return fmaxf(a, b); });
  max_val = warp::broadcast<constant::WarpSize>(max_val);

  float exp_sum = 0.f;
  for (int i = 0; i < size(regs); i++) {
    regs(i) = __expf(regs(i) - max_val);
    exp_sum += regs(i);
  }
  exp_sum = warp::reduce<constant::WarpSize>(exp_sum, [](float a, float b) { return a + b; });
  exp_sum = warp::broadcast<constant::WarpSize>(exp_sum);

  const float inv_sum = __frcp_rn(exp_sum + 1e-6f);
  CUTE_UNROLL
  for (int i = 0; i < size(regs); i++) {
    logits(i) = regs(i) * inv_sum;  // if predicate is false, will be set to 0
  }

  if (max_sum != nullptr && lane_idx == 0) {
    *max_sum = float2{max_val, exp_sum};
  }
}
#endif

#define BROADCAST0_BUFFER_SIZE_IN_BYTES 128

template <typename T, typename Func>
__forceinline__ __device__ T
broadcast0(void* buffer, Func&& value_producer) {
  if (threadIdx.x == 0) {
    *static_cast<T*>(buffer) = value_producer();
  }
  __syncthreads();
  return *static_cast<T*>(buffer);
}

template <int NumThreads, int HeadSize>
struct FlashAccCta {
  __forceinline__ __device__ static void
  acc(
      float* acc, float2* acc_max_sum,             // accumulative part
      const float* inc, const float2* inc_max_sum  // incremental part
  ) {
    float prev_max = acc_max_sum->x;
    float prev_sum = acc_max_sum->y;

    float curr_max = inc_max_sum->x;
    float curr_sum = inc_max_sum->y;

    float new_max = max(prev_max, curr_max);

    float prev_factor = new_max == prev_max ? 1.0f : __expf(prev_max - new_max);
    float curr_factor = new_max == curr_max ? 1.0f : __expf(curr_max - new_max);

    float new_sum = prev_factor * prev_sum + curr_factor * curr_sum;
    float new_sum_inv = 1.0f / new_sum;
    __syncthreads();  // ensure prev_max, prev_sum loaded
    if (threadIdx.x == 0) {
      *acc_max_sum = float2{new_max, new_sum};
    }

    bool load_prev_acc = prev_sum != 0.0f;
    CUTE_UNROLL
    for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
      // NOTE: In flash attention paper, curr_sum is not included, because it does not apply the denominator.
      // curr_sum cancels out the denominator for our implementation
      float old_val = load_prev_acc ? acc[i] : 0.0f;
      float new_val = prev_factor * (prev_sum * new_sum_inv) * old_val +
                      curr_factor * (curr_sum * new_sum_inv) * inc[i];
      // if (new_val != new_val) {
      //   printf(
      //       "acc[%d] new_val:%f new_sum:%f prev_factor:%f prev_sum:%f curr_factor:%f curr_sum:%f %f %f\n",
      //       i, new_val, new_sum, prev_factor, prev_sum, curr_factor, curr_sum, acc[i], inc[i]
      //   );
      // }
      acc[i] = new_val;
    }
  }

  template <typename TO>
  __forceinline__ __device__ static void
  atomic_acc(
      void* broadcast_buffer,
      volatile TO* acc, float2* acc_max_sum,
      float* inc, float2* inc_max_sum
  ) {
    float2 old = broadcast0<float2>(broadcast_buffer, [&]() {
      auto acc_max_sum_ull = reinterpret_cast<unsigned long long*>(acc_max_sum);
      constexpr uint64_t lock_bits = 0xFFFF'F000'FFF0'0000;  // float2{NaN, NaN};
      uint64_t old, assumed = lock_bits;
      bool locked = false;
      do {
        if (assumed == lock_bits) {
          // assumed = volatile_load(gmem_max_sum_ull);
          // __threadfence();
          assumed = atomicAdd(acc_max_sum_ull, 0);
          locked = false;
          continue;
        }
        old = atomicCAS(acc_max_sum_ull, assumed, lock_bits);
        locked = old == assumed;
        if (!locked) {
          assumed = old;
          backoff();
        }
      } while (!locked);
      union {
        float2 f32x2;
        uint64_t packed;
      };
      packed = old;
      return f32x2;
    });
    __syncthreads();
    __threadfence();

    float prev_max = old.x;
    float prev_sum = old.y;

    float curr_max = inc_max_sum->x;
    float curr_sum = inc_max_sum->y;

    float new_max = max(prev_max, curr_max);

    float prev_factor = new_max == prev_max ? 1.0f : __expf(prev_max - new_max);
    float curr_factor = new_max == curr_max ? 1.0f : __expf(curr_max - new_max);

    float new_sum = prev_factor * prev_sum + curr_factor * curr_sum;
    float new_sum_inv = 1.0f / new_sum;

    bool load_prev_acc = prev_sum != 0.0f;
    CUTE_UNROLL
    for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
      float old_val = load_prev_acc ? type_convert<float>(volatile_load(acc + i)) : 0.0f;
      float new_val = prev_factor * (prev_sum * new_sum_inv) * old_val +
                      curr_factor * (curr_sum * new_sum_inv) * inc[i];
      // if (new_val != new_val) {
      //   printf(
      //       "acc[%d] new_val:%f new_sum:%f prev_factor:%f prev_sum:%f curr_factor:%f curr_sum:%f %f %f\n",
      //       i, new_val, new_sum, prev_factor, prev_sum, curr_factor, curr_sum, old_val, inc[i]
      //   );
      // }
      acc[i] = type_convert<TO>(new_val);
    }

    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
      union {
        float2 f32x2;
        uint64_t packed;
      };
      f32x2 = float2{new_max, new_sum};
      atomicExch(reinterpret_cast<unsigned long long*>(acc_max_sum), packed);
    }
  }

  template <typename TO>
  __forceinline__ __device__ static void
  write_inc(
      TO* gmem, float2* gmem_max_sum,
      float* inc, float2* inc_max_sum
  ) {
    if (gmem_max_sum != nullptr && threadIdx.x == 0) {
      *gmem_max_sum = *inc_max_sum;
    }
    CUTE_UNROLL
    for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
      gmem[i] = type_convert<TO>(inc[i]);
    }
  }
};

template <int NumThreads, int HeadSize>
struct FlashAccWarp {
  __forceinline__ __device__ static void
  acc(
      float* acc, float2* acc_max_sum,             // accumulative part
      const float* inc, const float2* inc_max_sum  // incremental part
  ) {
    float prev_max = acc_max_sum->x;
    float prev_sum = acc_max_sum->y;

    float curr_max = inc_max_sum->x;
    float curr_sum = inc_max_sum->y;

    float new_max = max(prev_max, curr_max);

    float prev_factor = new_max == prev_max ? 1.0f : __expf(prev_max - new_max);
    float curr_factor = new_max == curr_max ? 1.0f : __expf(curr_max - new_max);

    float new_sum = prev_factor * prev_sum + curr_factor * curr_sum;
    float new_sum_inv = 1.0f / new_sum;
    if (lane_id() == 0) {
      *acc_max_sum = float2{new_max, new_sum};
    }

    bool load_prev_acc = prev_sum != 0.0f;
    CUTE_UNROLL
    for (int i = lane_id(); i < HeadSize; i += constant::WarpSize) {
      // NOTE: In flash attention paper, curr_sum is not included, because it does not apply the denominator.
      // curr_sum cancels out the denominator for our implementation
      float old_val = load_prev_acc ? acc[i] : 0.0f;
      float new_val = prev_factor * (prev_sum * new_sum_inv) * old_val +
                      curr_factor * (curr_sum * new_sum_inv) * inc[i];
      // if (new_val != new_val) {
      //   printf(
      //       "acc[%d] new_val:%f new_sum:%f prev_factor:%f prev_sum:%f curr_factor:%f curr_sum:%f %f %f\n",
      //       i, new_val, new_sum, prev_factor, prev_sum, curr_factor, curr_sum, acc[i], inc[i]
      //   );
      // }
      acc[i] = new_val;
    }
  }

  template <typename TO>
  __forceinline__ __device__ static void
  write_inc(
      TO* gmem, float2* gmem_max_sum,
      float* inc, float2* inc_max_sum
  ) {
    if (gmem_max_sum != nullptr && lane_id() == 0) {
      *gmem_max_sum = *inc_max_sum;
    }
    CUTE_UNROLL
    for (int i = lane_id(); i < HeadSize; i += constant::WarpSize) {
      gmem[i] = type_convert<TO>(inc[i]);
    }
  }
};

}  // namespace onnxruntime::contrib::paged
