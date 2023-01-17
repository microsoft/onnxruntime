// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_softmax_impl.h"

#include <limits>
#include <algorithm>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"
#include "core/providers/cuda/math/softmax_common.h"
#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Duplicated softmax_impl.cu here
// So far attempt to use shared kernel with additional template resulted in lost performance

// Note: The intended case for 'input_bias' is the input sequence mask for transformer models
// As an additive mask, it should be zero for preserved tokens and -infty for tokens to screen
// The mask will broadcast from [batch_size, 1, 1, seq_len] to input [batch_size, num_heads, seq_len, seq_len]
// Here element_count = seq_len and bias_broadcast_size_per_batch = num_heads * seq_len

// The softmax + additive mask fusion follows NVIDIA apex's additive_masked_softmax_warp_forward
// see
// https://github.com/NVIDIA/apex/blob/4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a/apex/contrib/csrc/multihead_attn/softmax.h

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_inner_broadcast>
__global__ void BiasSoftmaxWarpForward(output_t* output, const input_t* input, const input_t* input_bias,
                                       int element_count, int batch_count, fast_divmod bias_broadcast_fdm) {
  // "WARP" refers to cooperative threads and might not equal 32 threads of GPU warp
  // thread block is (WARP_SIZE, 128/WARP_SIZE)
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = next_power_of_two < GPU_WARP_SIZE ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
#ifdef USE_ROCM
  constexpr int WARP_BATCH = 1;
#else
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
#endif

  // each "WARP" (<=32) processes WARP_BATCH(one of {1,2}) batches
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // last warp may have fewer batches
  int local_batches = batch_count - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;

  // thread will process elements (local_index + n * warp_size) within batch
  int local_idx = threadIdx.x;

  // push input, input_bias output pointers to batch we need to process
  input += first_batch * element_count + local_idx;
  output += first_batch * element_count + local_idx;

  // load from global memory and apply bias (likely an additive mask)
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    // If is_inner_broadcast, input shape is [x, broadcast_size, element_count], bias shape is [x, 1, element_count].
    // Otherwise, input shape is [x, broadcast_size, element_count], bias shape is [1, broadcast_size, element_count].
    int bias_batch_offset =
        is_inner_broadcast ? bias_broadcast_fdm.div(first_batch + i) : bias_broadcast_fdm.mod(first_batch + i);
    int bias_offset = bias_batch_offset * element_count + local_idx;
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] =
            (acc_t)input[i * element_count + it * WARP_SIZE] + (acc_t)input_bias[bias_offset + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // find maximum value within batch for numerical stability
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

  // normalization factor Z = Sum[ exp(element_i), for element_i in batch ]
  acc_t sum[WARP_BATCH]{acc_t(0.0)};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp((acc_t)(elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// write back normalized value = exp(element_i)/Z to global memory
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        output[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T>
Status BiasSoftmaxImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* output_data, const T* input_data,
                       const T* bias_data, int element_count, int batch_count, bool is_inner_broadcast,
                       int bias_broadcast_size) {
  if (element_count == 0) return Status::OK();
  if (element_count <= 1024 && element_count * static_cast<int>(sizeof(T)) <= 4096) {
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

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    fast_divmod bias_broadcast_fdm = fast_divmod(bias_broadcast_size);

    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
#define LAUNCHE_BIAS_SOFTMAX_KERNEL(log2_elements_value, is_inner_broadcast_value)                                   \
  BiasSoftmaxWarpForward<T, T, AccT, log2_elements_value, is_inner_broadcast_value><<<blocks, threads, 0, stream>>>( \
      output_data, input_data, bias_data, element_count, batch_count, bias_broadcast_fdm)
#define CASE_LOG2_ELEMENTS(log2_elements_value)                \
  case log2_elements_value: {                                  \
    if (is_inner_broadcast) {                                  \
      LAUNCHE_BIAS_SOFTMAX_KERNEL(log2_elements_value, true);  \
    } else {                                                   \
      LAUNCHE_BIAS_SOFTMAX_KERNEL(log2_elements_value, false); \
    }                                                          \
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
#undef CASE_LOG2_ELEMENTS
#undef LAUNCHE_BIAS_SOFTMAX_KERNEL
    }
    return Status::OK();
  }

  // For large element count we fall back to explicit Add kernel + CUDA DNN library
  // note: This is an unhappy path! There is no performance benefit for the fusion.
  int output_rank_or_simple_broadcast = 3;
  TArray<int64_t> rhs_strides;
  TArray<fast_divmod> output_fdms;
  const TArray<int64_t>* p_rhs_strides = nullptr;
  const TArray<fast_divmod>* p_output_fdms = nullptr;
  fast_divmod fdm_h(1);
  fast_divmod fdm_c;
  if ((is_inner_broadcast && bias_broadcast_size == 1) || (!is_inner_broadcast && bias_broadcast_size == batch_count)) {
    // input and bias shape is same.
    output_rank_or_simple_broadcast = static_cast<int>(SimpleBroadcast::NoBroadcast);
  } else if (!is_inner_broadcast) {
    output_rank_or_simple_broadcast = static_cast<int>(SimpleBroadcast::RightPerChannelBatchN);
    fdm_c = fast_divmod(element_count * bias_broadcast_size);
  } else {
    rhs_strides.SetSize(3);
    rhs_strides[0] = static_cast<int64_t>(element_count);
    rhs_strides[1] = 0LL;
    rhs_strides[2] = 1LL;
    p_rhs_strides = &rhs_strides;
    output_fdms.SetSize(3);
    output_fdms[0] = fast_divmod(element_count * bias_broadcast_size);
    output_fdms[1] = fast_divmod(element_count);
    output_fdms[2] = fast_divmod(1);
    p_output_fdms = &output_fdms;
  }

  BinaryElementWiseImpl(stream, output_rank_or_simple_broadcast, nullptr, input_data, p_rhs_strides, bias_data,
                        p_output_fdms, fdm_h, fdm_c, output_data, OP_Add<T, T, T>(),
                        static_cast<size_t>(batch_count * element_count));

  // invoke cuda DNN library for Y = softmax(X)
  const int64_t dims[]{batch_count, 1, 1, element_count};
  const auto alpha = Consts<T>::One;
  const auto beta = Consts<T>::Zero;
  CudnnTensor input_tensor, output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<T>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<T>()));
  return SoftmaxForward(cudnn_handle, &alpha, input_tensor, output_data, &beta, output_tensor, output_data);
}

#define SPECIALIZED_BIAS_SOFTMAX_IMPL(T)                                                                          \
  template Status BiasSoftmaxImpl<T>(cudaStream_t stream, cudnnHandle_t cudnn_handle, T * output_data,            \
                                     const T* input_data, const T* bias_data, int element_count, int batch_count, \
                                     bool is_inner_broadcast, int bias_broadcast_size);

// MIOpen doesn't support double so ROCm kernel doesn't have double support for now.
SPECIALIZED_BIAS_SOFTMAX_IMPL(float)
SPECIALIZED_BIAS_SOFTMAX_IMPL(half)
#ifdef USE_CUDA
SPECIALIZED_BIAS_SOFTMAX_IMPL(double)
#endif

#undef SPECIALIZED_BIAS_SOFTMAX_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
