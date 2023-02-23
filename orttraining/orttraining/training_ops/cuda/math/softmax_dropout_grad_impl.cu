// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_dropout_grad_impl.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "orttraining/training_ops/cuda/math/softmax_grad_impl.h"
#include "orttraining/training_ops/cuda/nn/dropout_grad_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename AccT, int Log2Elements, int NumUnroll>
__global__ void SoftmaxDropoutGradKernel(T* input_grad_data, const T* output_grad_data, const bool* mask_data,
                                         const T* softmax_output_data, int element_count, int batch_count,
                                         const float scale) {
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
  int local_idx = threadIdx.x;
  int thread_offset = first_batch * element_count + NumUnroll * local_idx;
  const T* thread_output_grad_data = output_grad_data + thread_offset;
  const bool* thread_mask_data = mask_data + thread_offset;
  const T* thread_softmax_output_data = softmax_output_data + thread_offset;

  using TVec = aligned_vector<T, NumUnroll>;
  using MaskVec = aligned_vector<bool, NumUnroll>;
  const T zero = static_cast<T>(0.0f);

  // load data from global memory
  T output_grad_value_t[kWarpBatch][kWarpIterations];
  bool mask_value[kWarpBatch][kWarpIterations];
  T softmax_output_value_t[kWarpBatch][kWarpIterations];

#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < kWarpIterations; it += NumUnroll) {
      int itr_jmp = it * kWarpSize;
      int element_index = itr_jmp + NumUnroll * local_idx;
      if (element_index < batch_element_count) {
        itr_jmp += (i * element_count);
        if (NumUnroll > 1) {
          *(reinterpret_cast<TVec*>(&output_grad_value_t[i][it])) =
              *(reinterpret_cast<const TVec*>(thread_output_grad_data + itr_jmp));
          *(reinterpret_cast<MaskVec*>(&mask_value[i][it])) =
              *(reinterpret_cast<const MaskVec*>(thread_mask_data + itr_jmp));
          *(reinterpret_cast<TVec*>(&softmax_output_value_t[i][it])) =
              *(reinterpret_cast<const TVec*>(thread_softmax_output_data + itr_jmp));
        } else {
          output_grad_value_t[i][it] = thread_output_grad_data[itr_jmp];
          mask_value[i][it] = thread_mask_data[itr_jmp];
          softmax_output_value_t[i][it] = thread_softmax_output_data[itr_jmp];
        }
      } else {
#pragma unroll
        for (int element = 0; element < NumUnroll; ++element) {
          output_grad_value_t[i][it + element] = zero;
          mask_value[i][it + element] = false;
          softmax_output_value_t[i][it + element] = zero;
        }
      }
    }
  }

  AccT output_grad_value_acct[kWarpBatch][kWarpIterations];
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
    for (int it = 0; it < kWarpIterations; ++it) {
      output_grad_value_acct[i][it] =
          static_cast<AccT>(output_grad_value_t[i][it] * softmax_output_value_t[i][it]) * mask_value[i][it] * scale;
    }
  }

  AccT sum[kWarpBatch];
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
    sum[i] = output_grad_value_acct[i][0];
#pragma unroll
    for (int it = 1; it < kWarpIterations; ++it) {
      sum[i] += output_grad_value_acct[i][it];
    }
  }

  // reduction sum
  warp_reduce<AccT, kWarpBatch, kWarpSize, Add>(sum);

  T input_grad_data_t[kWarpBatch][kWarpIterations];
#pragma unroll
  for (int i = 0; i < kWarpBatch; ++i) {
#pragma unroll
    for (int it = 0; it < kWarpIterations; ++it) {
      input_grad_data_t[i][it] =
          static_cast<T>(output_grad_value_acct[i][it] - sum[i] * static_cast<AccT>(softmax_output_value_t[i][it]));
    }
  }

  // store result
  T* thread_input_grad_data = input_grad_data + thread_offset;
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
          *(reinterpret_cast<TVec*>(thread_input_grad_data + itr_jmp)) =
              *(reinterpret_cast<TVec*>(&input_grad_data_t[i][it]));
        } else {
          thread_input_grad_data[itr_jmp] = input_grad_data_t[i][it];
        }
      }
    }
  }
}

template <typename T>
Status SoftmaxDropoutGradImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* input_grad_data,
                              const T* output_grad_data, const bool* mask_data, const T* softmax_output_data,
                              int element_count, int batch_count, const float ratio) {
  if (element_count == 0) return Status::OK();
  if (element_count <= 2048) {
    typedef AccumulationType_t<T> AccT;
    const float scale = 1.f / (1.f - ratio);
    int log2_elements = log2_ceil(element_count);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = std::min(next_power_of_two, GPU_WARP_SIZE_HOST);

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
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
                     reinterpret_cast<uint64_t>(input_grad_data) % t_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(output_grad_data) % t_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(mask_data) % mask_vec4_alignment == 0 &&
                     reinterpret_cast<uint64_t>(softmax_output_data) % t_vec4_alignment == 0;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    switch (log2_elements) {
#define LAUNCH_SOFTMAX_DROPOUT_GRAD_KERNEL(log2_elements_value, num_unroll)                           \
  SoftmaxDropoutGradKernel<T, AccT, log2_elements_value, num_unroll><<<blocks, threads, 0, stream>>>( \
      input_grad_data, output_grad_data, mask_data, softmax_output_data, element_count, batch_count, scale)
#define CASE_LOG2_ELEMENTS(log2_elements_value)                   \
  case log2_elements_value: {                                     \
    if (flag_vec4) {                                              \
      LAUNCH_SOFTMAX_DROPOUT_GRAD_KERNEL(log2_elements_value, 4); \
    } else {                                                      \
      LAUNCH_SOFTMAX_DROPOUT_GRAD_KERNEL(log2_elements_value, 1); \
    }                                                             \
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
#undef LAUNCH_SOFTMAX_DROPOUT_GRAD_KERNEL
    }

    return Status::OK();
  }

  // General solution to call DropoutGrad and SoftmaxGrad respectively.
  DropoutGradientKernelImpl(stream, static_cast<int64_t>(batch_count * element_count), output_grad_data, mask_data,
                            ratio, input_grad_data, false);
  return SoftmaxGradImpl(stream, cudnn_handle, input_grad_data, input_grad_data, softmax_output_data, element_count,
                         batch_count, false);
}

#define SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(T)                                                       \
  template Status SoftmaxDropoutGradImpl<T>(                                                           \
      cudaStream_t stream, cudnnHandle_t cudnn_handle, T * input_grad_data, const T* output_grad_data, \
      const bool* mask_data, const T* softmax_output_data, int element_count, int batch_count, const float ratio);

SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(half)
#ifdef USE_CUDA
SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL(double)
#endif

#undef SPECIALIZED_SOFTMAX_DROPOUT_GRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime
