// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/bias_softmax_dropout_impl.h"

#include <curand_kernel.h>
#include <algorithm>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/nn/dropout_impl.h"
#include "contrib_ops/cuda/math/bias_softmax_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename AccT, int Log2Elements, int NumUnroll, bool IsInnerBroadcast>
__global__ void BiasSoftmaxDropoutKernel(T* dropout_output_data, bool* mask_data, T* softmax_output_data,
                                         const T* input_data, const T* bias_data, int element_count, int batch_count,
                                         fast_divmod bias_broadcast_fdm, const float ratio,
                                         const std::pair<uint64_t, uint64_t> seeds) {
  constexpr int kNextPowOfTwo = 1 << Log2Elements;
  constexpr int kWarpSize = kNextPowOfTwo < GPU_WARP_SIZE ? kNextPowOfTwo : GPU_WARP_SIZE;
  constexpr int kWarpIterations = kNextPowOfTwo / kWarpSize;
#ifdef USE_ROCM
  constexpr int kWarpBatch = 1;
#else
  constexpr int kWarpBatch = (kNextPowOfTwo <= 128) ? 2 : 1;
#endif

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kWarpBatch;
  // last warp may have fewer batches.
  int local_batches = batch_count - first_batch;
  if (local_batches > kWarpBatch) local_batches = kWarpBatch;

  int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int local_idx = threadIdx.x;

  T input_value[kWarpBatch][kWarpIterations];
  T bias_value[kWarpBatch][kWarpIterations];

  int thread_offset = first_batch * element_count + NumUnroll * local_idx;
  const T* thread_input_data = input_data + thread_offset;

  using TVec = aligned_vector<T, NumUnroll>;
  using MaskVec = aligned_vector<bool, NumUnroll>;

#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
    // If IsInnerBroadcast, input shape is [x, broadcast_size, element_count], bias shape is [x, 1, element_count].
    // Otherwise, input shape is [x, broadcast_size, element_count], bias shape is [1, broadcast_size, element_count].
    int bias_batch_offset =
        IsInnerBroadcast ? bias_broadcast_fdm.div(first_batch + i) : bias_broadcast_fdm.mod(first_batch + i);
    const T* batch_bias_data = bias_data + bias_batch_offset * element_count + NumUnroll * local_idx;
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < kWarpIterations; it += NumUnroll) {
      int itr_jmp = it * kWarpSize;
      int element_index = itr_jmp + NumUnroll * local_idx;
      if (element_index < batch_element_count) {
        int itr_offset = i * element_count + itr_jmp;
        if (NumUnroll > 1) {
          *(reinterpret_cast<TVec*>(&input_value[i][it])) =
              *(reinterpret_cast<const TVec*>(thread_input_data + itr_offset));
          *(reinterpret_cast<TVec*>(&bias_value[i][it])) = *(reinterpret_cast<const TVec*>(batch_bias_data + itr_jmp));
        } else {
          input_value[i][it] = thread_input_data[itr_offset];
          bias_value[i][it] = batch_bias_data[itr_jmp];
        }
      } else {
#pragma unroll
        for (int element = 0; element < NumUnroll; ++element) {
          input_value[i][it + element] = static_cast<T>(-std::numeric_limits<AccT>::infinity());
          bias_value[i][it + element] = static_cast<T>(0.0f);
        }
      }
    }
  }

  AccT input_value_acct[kWarpBatch][kWarpIterations];
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
    for (int it = 0; it < kWarpIterations; ++it) {
      input_value_acct[i][it] = static_cast<AccT>(input_value[i][it] + bias_value[i][it]);
    }
  }

  // compute local max_value
  AccT max_value[kWarpBatch];
  AccT sum[kWarpBatch];
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
    max_value[i] = input_value_acct[i][0];
    sum[i] = 0.0f;
  }

#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
    for (int it = 1; it < kWarpIterations; ++it) {
      max_value[i] = (max_value[i] > input_value_acct[i][it]) ? max_value[i] : input_value_acct[i][it];
    }
  }

  // reduction max_value
  warp_reduce<AccT, kWarpBatch, kWarpSize, Max>(max_value);

#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
    for (int it = 0; it < kWarpIterations; ++it) {
      input_value_acct[i][it] = _Exp(input_value_acct[i][it] - max_value[i]);
      sum[i] += input_value_acct[i][it];
    }
  }

  // reduction sum
  warp_reduce<AccT, kWarpBatch, kWarpSize, Add>(sum);

  T dropout_output_value[kWarpBatch][kWarpIterations];
  bool mask_value[kWarpBatch][kWarpIterations];
  T softmax_output_value[kWarpBatch][kWarpIterations];

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  if (NumUnroll == 4) {
    float4 rand4;
#pragma unroll
    for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
      for (int it = 0; it < kWarpIterations; it += NumUnroll) {
        rand4 = curand_uniform4(&state);
#pragma unroll
        for (int element = 0; element < NumUnroll; ++element) {
          mask_value[i][it + element] = (&rand4.x)[element] < p;
          AccT value = input_value_acct[i][it + element] / sum[i];
          softmax_output_value[i][it + element] = static_cast<T>(value);
          dropout_output_value[i][it + element] = static_cast<T>(value * scale * mask_value[i][it + element]);
        }
      }
    }
  } else {
    float rand;
#pragma unroll
    for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
      for (int it = 0; it < kWarpIterations; ++it) {
        rand = curand_uniform(&state);
        mask_value[i][it] = rand < p;
        AccT value = input_value_acct[i][it] / sum[i];
        softmax_output_value[i][it] = static_cast<T>(value);
        dropout_output_value[i][it] = static_cast<T>(value * scale * mask_value[i][it]);
      }
    }
  }

  // store result
  T* thread_dropout_output_data = dropout_output_data + thread_offset;
  bool* thread_mask_data = mask_data + thread_offset;
  T* thread_softmax_output_data = softmax_output_data + thread_offset;
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < kWarpIterations; it += NumUnroll) {
      int itr_jmp = it * kWarpSize;
      int element_index = itr_jmp + NumUnroll * local_idx;
      if (element_index < element_count) {
        itr_jmp += (i * element_count);
        if (NumUnroll > 1) {
          *(reinterpret_cast<TVec*>(thread_dropout_output_data + itr_jmp)) =
              *(reinterpret_cast<TVec*>(&dropout_output_value[i][it]));
          *(reinterpret_cast<MaskVec*>(thread_mask_data + itr_jmp)) = *(reinterpret_cast<MaskVec*>(&mask_value[i][it]));
          *(reinterpret_cast<TVec*>(thread_softmax_output_data + itr_jmp)) =
              *(reinterpret_cast<TVec*>(&softmax_output_value[i][it]));
        } else {
          thread_dropout_output_data[itr_jmp] = dropout_output_value[i][it];
          thread_mask_data[itr_jmp] = mask_value[i][it];
          thread_softmax_output_data[itr_jmp] = softmax_output_value[i][it];
        }
      }
    }
  }
}

template <typename T>
Status BiasSoftmaxDropoutImpl(cudaStream_t stream, const cudaDeviceProp& prop, cudnnHandle_t cudnn_handle,
                              T* dropout_output_data, bool* mask_data, T* softmax_output_data, const T* input_data,
                              const T* bias_data, int element_count, int batch_count, bool is_inner_broadcast,
                              int bias_broadcast_size, const float ratio, PhiloxGenerator& generator) {
  if (element_count == 0) return Status::OK();
  if (element_count <= 2048) {
    typedef AccumulationType_t<T> AccT;
    int log2_elements = log2_ceil(element_count);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = std::min(next_power_of_two, GPU_WARP_SIZE_HOST);

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
#ifdef USE_ROCM
    int batches_per_warp = 1;
    constexpr int threads_per_block = 256;
#else
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int threads_per_block = 128;
#endif

    constexpr int t_vec4_alignment = std::alignment_of<aligned_vector<T, 4>>::value;
    constexpr int mask_vec4_alignment = std::alignment_of<aligned_vector<bool, 4>>::value;
    bool flag_vec4 = element_count % 4 == 0 && next_power_of_two / warp_size >= 4 &&
                     reinterpret_cast<uint64_t>(dropout_output_data) % t_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(mask_data) % mask_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(softmax_output_data) % t_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(input_data) % t_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(bias_data) % t_vec4_alignment == 0;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    fast_divmod bias_broadcast_fdm = fast_divmod(bias_broadcast_size);
    auto seeds = generator.NextPhiloxSeeds(static_cast<uint64_t>((next_power_of_two / warp_size) * batches_per_warp));

    switch (log2_elements) {
#define LAUNCH_BIAS_SOFTMAX_DROPOUT_KERNEL(log2_elements_value, num_unroll, is_inner_broadcast_value)              \
  BiasSoftmaxDropoutKernel<T, AccT, log2_elements_value, num_unroll, is_inner_broadcast_value>                     \
      <<<blocks, threads, 0, stream>>>(dropout_output_data, mask_data, softmax_output_data, input_data, bias_data, \
                                       element_count, batch_count, bias_broadcast_fdm, ratio, seeds)
#define HANDLE_IS_INNER_BROADCAST(log2_elements_value, num_unroll)              \
  if (is_inner_broadcast) {                                                     \
    LAUNCH_BIAS_SOFTMAX_DROPOUT_KERNEL(log2_elements_value, num_unroll, true);  \
  } else {                                                                      \
    LAUNCH_BIAS_SOFTMAX_DROPOUT_KERNEL(log2_elements_value, num_unroll, false); \
  }
#define CASE_LOG2_ELEMENTS(log2_elements_value)         \
  case log2_elements_value: {                           \
    if (flag_vec4) {                                    \
      HANDLE_IS_INNER_BROADCAST(log2_elements_value, 4) \
    } else {                                            \
      HANDLE_IS_INNER_BROADCAST(log2_elements_value, 1) \
    }                                                   \
  } break
      CASE_LOG2_ELEMENTS(0);   // 1
      CASE_LOG2_ELEMENTS(1);   // 2
      CASE_LOG2_ELEMENTS(2);   // 4
      CASE_LOG2_ELEMENTS(3);   // 8
      CASE_LOG2_ELEMENTS(4);   // 16
      CASE_LOG2_ELEMENTS(5);   // 32
      CASE_LOG2_ELEMENTS(6);   // 64
      CASE_LOG2_ELEMENTS(7);   // 128
      CASE_LOG2_ELEMENTS(8);   // 256
      CASE_LOG2_ELEMENTS(9);   // 512
      CASE_LOG2_ELEMENTS(10);  // 1024
      CASE_LOG2_ELEMENTS(11);  // 2048
#undef CASE_LOG2_ELEMENTS
#undef HANDLE_IS_INNER_BROADCAST
#undef LAUNCH_BIAS_SOFTMAX_DROPOUT_KERNEL
    }

    return Status::OK();
  }

  // General solution to call BiasSoftmax and Dropout respectively.
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::BiasSoftmaxImpl(stream, cudnn_handle, softmax_output_data, input_data,
                                                                  bias_data, element_count, batch_count,
                                                                  is_inner_broadcast, bias_broadcast_size));
  DropoutKernelImpl(prop, stream, static_cast<int64_t>(batch_count * element_count), 0LL, ratio, generator,
                    softmax_output_data, dropout_output_data, mask_data, false);
  return Status::OK();
}

#define SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(T)                                                                 \
  template Status BiasSoftmaxDropoutImpl<T>(cudaStream_t stream, const cudaDeviceProp& prop,                     \
                                            cudnnHandle_t cudnn_handle, T* dropout_output_data, bool* mask_data, \
                                            T* softmax_output_data, const T* input_data, const T* bias_data,     \
                                            int element_count, int batch_count, bool is_inner_broadcast,         \
                                            int bias_broadcast_size, const float ratio, PhiloxGenerator& generator);

SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(float)
SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(half)
#ifdef USE_CUDA
SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL(double)
#endif

#undef SPECIALIZED_BIAS_SOFTMAX_DROPOUT_IMPL

}  // namespace cuda
}  // namespace onnxruntime
