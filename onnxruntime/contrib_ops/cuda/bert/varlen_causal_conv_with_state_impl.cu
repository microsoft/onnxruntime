// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Packed variable-length causal depthwise convolution. The general path assigns
// one block to a (request, channel tile), so warp lanes access token-major rows
// contiguously while request boundaries and state ownership remain isolated.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "contrib_ops/cuda/bert/varlen_causal_conv_with_state.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

__device__ __forceinline__ float VarlenSilu(float x) {
  return x / (1.0f + expf(-x));
}

// The all-one path still reads and validates the exact row [b, b + 1]. This is
// important when N == B but malformed offsets do not actually describe B
// one-token sequences.
template <typename T>
__global__ void VarlenCausalConvDecodeKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* initial_state,
    T* __restrict__ output,
    T* final_state,
    T* state_update,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ capture_count,
    int batch_channels,
    int batch_size,
    int total_tokens,
    int channels,
    int kernel_size,
    int dilation,
    bool apply_silu,
    int state_update_capacity) {
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) {
    return;
  }

  const int b = bc / channels;
  const int c = bc % channels;
  const int32_t first = cu_seqlens[0];
  const int32_t last = cu_seqlens[batch_size];
  const int32_t start = cu_seqlens[b];
  const int32_t end = cu_seqlens[b + 1];
  if (first != 0 || last != total_tokens || start != b || end != b + 1) {
    return;
  }

  const int pad = (kernel_size - 1) * dilation;
  const int64_t state_offset = static_cast<int64_t>(bc) * pad;
  const int64_t weight_offset = static_cast<int64_t>(c) * kernel_size;
  const T input_value = input[static_cast<int64_t>(b) * channels + c];
  if (state_update != nullptr && capture_count[b] > 0) {
    state_update[static_cast<int64_t>(b) * state_update_capacity * channels + c] = input_value;
  }

  float sum = bias != nullptr ? to_float(bias[c]) : 0.0f;
  if (pad == 0) {
    // K=1 has no state storage and therefore no aliasing access.
    sum += to_float(weight[weight_offset]) * to_float(input_value);
  } else {
    // Compute the complete result from the old state before shifting in ascending order. Every
    // destination is written only after its source was consumed by the dot product, and
    // state[k + 1] is read before state[k] is overwritten, so initial_state == final_state is
    // explicitly safe. Tap k reads the state element dilation positions apart; the newest tap
    // (k = kernel_size - 1) is the incoming token itself.
    //
    // The state stores every one of the last pad raw samples, not only the dilated tap positions,
    // so the shift is always by one sample regardless of dilation: a slot that is between two taps
    // now becomes a tap slot on a later token. Shifting by dilation would drop those samples.
    for (int k = 0; k < kernel_size - 1; ++k) {
      sum += to_float(weight[weight_offset + k]) *
             to_float(initial_state[state_offset + static_cast<int64_t>(k) * dilation]);
    }
    sum += to_float(weight[weight_offset + kernel_size - 1]) * to_float(input_value);
    for (int k = 0; k < pad - 1; ++k) {
      final_state[state_offset + k] = initial_state[state_offset + k + 1];
    }
    final_state[state_offset + pad - 1] = input_value;
  }

  if (apply_silu) {
    sum = VarlenSilu(sum);
  }
  output[static_cast<int64_t>(b) * channels + c] = from_float<T>(sum);
}

template <typename T>
__device__ __forceinline__ T ReadStateOrInput(
    const T* __restrict__ input,
    const T* staged_state,
    int32_t start,
    int channels,
    int c,
    int pad,
    int local_position) {
  if (local_position < 0) {
    return staged_state[local_position + pad];
  }
  return input[(static_cast<int64_t>(start) + local_position) * channels + c];
}

// General ragged path. Each thread owns one channel and processes all tokens in
// the request serially. Consequently, lanes in a warp read and write contiguous
// channels of each packed token row instead of striding across rows by C.
//
// Dynamic shared memory contains the complete old K-1 state for every channel
// in the tile. All active threads finish staging and the complete block
// synchronizes before any output or final-state write. A channel
// belongs to exactly one tile, so this also makes initial_state == final_state
// safe without synchronization between blocks.
template <typename T>
__global__ void VarlenCausalConvKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* initial_state,
    T* __restrict__ output,
    T* final_state,
    T* state_update,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ capture_count,
    int batch_size,
    int total_tokens,
    int channels,
    int kernel_size,
    int dilation,
    bool apply_silu,
    int state_update_capacity) {
  const int tid = threadIdx.x;
  const int channel_tiles = static_cast<int>(
      (static_cast<int64_t>(channels) + blockDim.x - 1) / blockDim.x);
  const int64_t linear_block = static_cast<int64_t>(blockIdx.x);
  const int b = static_cast<int>(linear_block / channel_tiles);
  const int channel_tile = static_cast<int>(linear_block % channel_tiles);
  const int64_t tile_first_channel = static_cast<int64_t>(channel_tile) * blockDim.x;
  const int64_t c64 = tile_first_channel + tid;
  const bool channel_active = c64 < channels;
  const int c = channel_active ? static_cast<int>(c64) : 0;
  const int pad = (kernel_size - 1) * dilation;

  // These values are block-uniform. A malformed interval returns the complete
  // block before any input, state, or output access.
  const int32_t first = cu_seqlens[0];
  const int32_t last = cu_seqlens[batch_size];
  const int32_t start = cu_seqlens[b];
  const int32_t end = cu_seqlens[b + 1];
  if (first != 0 || last != total_tokens ||
      start < 0 || start >= end || end > total_tokens) {
    return;
  }
  const int local_length = end - start;
  const int captured = state_update == nullptr
                           ? 0
                           : max(0, min(min(capture_count[b], state_update_capacity), local_length));

  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* staged_state = reinterpret_cast<T*>(shared_bytes);
  T* channel_state = staged_state + static_cast<int64_t>(tid) * pad;
  const int64_t state_offset =
      (static_cast<int64_t>(b) * channels + c) * pad;
  if (pad > 0) {
    // Cooperatively copy the channel-major state tile as one contiguous range.
    // channel_state remains the private per-thread view after the barrier.
    const int active_channels =
        min(static_cast<int>(blockDim.x), channels - static_cast<int>(tile_first_channel));
    const int64_t staged_elements = static_cast<int64_t>(active_channels) * pad;
    const int64_t state_tile_offset =
        (static_cast<int64_t>(b) * channels + tile_first_channel) * pad;
    // Keep this read in the same block that writes final_state below. CUDA
    // provides no cross-block ordering, so splitting token work across blocks
    // would race when initial_state and final_state alias.
    for (int64_t i = tid; i < staged_elements; i += blockDim.x) {
      staged_state[i] = initial_state[state_tile_offset + i];
    }
    __syncthreads();
  }

  if (!channel_active) {
    return;
  }

  const int64_t weight_offset = static_cast<int64_t>(c) * kernel_size;
  const float bias_value = bias != nullptr ? to_float(bias[c]) : 0.0f;
  for (int t = 0; t < local_length; ++t) {
    float sum = bias_value;
    for (int k = 0; k < kernel_size; ++k) {
      sum += to_float(weight[weight_offset + k]) *
             to_float(ReadStateOrInput(input, channel_state, start, channels, c, pad,
                                       t - pad + k * dilation));
    }
    if (apply_silu) {
      sum = VarlenSilu(sum);
    }
    output[(static_cast<int64_t>(start) + t) * channels + c] = from_float<T>(sum);
    if (t < captured) {
      state_update[(static_cast<int64_t>(b) * state_update_capacity + t) * channels + c] =
          input[(static_cast<int64_t>(start) + t) * channels + c];
    }
  }

  if (pad == 0) {
    return;
  }

  // final_state is always fully written, including when local_length < pad.
  // This block staged the corresponding initial_state tile before any write,
  // which is required by the documented same-allocation contract.
  for (int k = 0; k < pad; ++k) {
    final_state[state_offset + k] =
        ReadStateOrInput(input, channel_state, start, channels, c, pad, local_length - pad + k);
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* bias,
    const T* initial_state,
    T* output,
    T* final_state,
    T* state_update,
    const int32_t* cu_seqlens,
    const int32_t* capture_count,
    int batch_size,
    int total_tokens,
    bool all_ones,
    int channels,
    int kernel_size,
    int dilation,
    bool apply_silu,
    int max_threads_per_block,
    int state_update_capacity) {
  if (all_ones) {
    const int64_t batch_channels = static_cast<int64_t>(batch_size) * channels;
    ORT_RETURN_IF_NOT(batch_channels <= std::numeric_limits<int>::max(),
                      "VarlenCausalConvWithState: batch_size * channels exceeds INT_MAX");
    const int threads = std::min(256, max_threads_per_block);
    const int blocks = static_cast<int>((batch_channels + threads - 1) / threads);
    VarlenCausalConvDecodeKernel<T><<<blocks, threads, 0, stream>>>(
        input, weight, bias, initial_state, output, final_state, state_update,
        cu_seqlens, capture_count, static_cast<int>(batch_channels), batch_size, total_tokens,
        channels, kernel_size, dilation, apply_silu, state_update_capacity);
    return CUDA_CALL(cudaGetLastError());
  }

  constexpr size_t kMaxStagedStateBytes = 48 * 1024;
  const size_t state_bytes_per_channel =
      static_cast<size_t>(kernel_size - 1) * static_cast<size_t>(dilation) * sizeof(T);
  // Stay within CUDA's portable 48-KiB per-block shared-memory budget instead
  // of requiring a device-specific dynamic-shared-memory opt-in. Practical
  // Qwen kernels use only a few state elements per channel and retain the full
  // 256-channel tile. For unusually wide kernels, reduce the tile; reject only
  // when one channel's old state alone cannot fit.
  ORT_RETURN_IF_NOT(state_bytes_per_channel <= kMaxStagedStateBytes,
                    "VarlenCausalConvWithState: one channel's kernel state requires more than 48 KiB");

  int threads_limit = std::min({256, max_threads_per_block, channels});
  if (state_bytes_per_channel > 0) {
    threads_limit = std::min(
        threads_limit, static_cast<int>(kMaxStagedStateBytes / state_bytes_per_channel));
  }
  ORT_RETURN_IF_NOT(threads_limit > 0,
                    "VarlenCausalConvWithState: no channel state fits in the shared-memory budget");

  int threads = threads_limit;
  if (threads_limit >= 32) {
    const int warp_aligned_limit = (threads_limit / 32) * 32;
    threads = channels < 32
                  ? 32
                  : std::min(warp_aligned_limit, ((channels + 31) / 32) * 32);
  }

  const int64_t channel_tiles = (static_cast<int64_t>(channels) + threads - 1) / threads;
  const int64_t general_blocks = static_cast<int64_t>(batch_size) * channel_tiles;
  ORT_RETURN_IF_NOT(general_blocks <= std::numeric_limits<int>::max(),
                    "VarlenCausalConvWithState: general kernel grid exceeds INT_MAX blocks");
  const size_t shared_memory_bytes = static_cast<size_t>(threads) * state_bytes_per_channel;
  VarlenCausalConvKernel<T><<<static_cast<unsigned int>(general_blocks), threads,
                              shared_memory_bytes, stream>>>(
      input, weight, bias, initial_state, output, final_state, state_update,
      cu_seqlens, capture_count, batch_size, total_tokens, channels, kernel_size, dilation,
      apply_silu, state_update_capacity);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchVarlenCausalConvWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, float*, const int32_t*, const int32_t*,
    int, int, bool, int, int, int, bool, int, int);

template Status LaunchVarlenCausalConvWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    half*, half*, half*, const int32_t*, const int32_t*,
    int, int, bool, int, int, int, bool, int, int);

template Status LaunchVarlenCausalConvWithStateKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    const int32_t*, const int32_t*, int, int, bool, int, int, int, bool, int, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
