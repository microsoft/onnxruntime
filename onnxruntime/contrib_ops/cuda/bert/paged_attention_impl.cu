// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <type_traits>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include <cublas_v2.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////// Quantized paged KV cache helpers
//
// Symmetric, zero-point-free quantization with the same numerics GroupQueryAttention uses
// (group_query_attention_qdq.cuh): INT8 rounds to nearest and clamps to [-128, 127]; FP8 E4M3
// clamps to +/-448 and lets the hardware convert. The paged layout
// [num_blocks, block_size, kv_num_heads, head_size] makes the scale index trivial: the innermost
// (kv_head, channel) pair is exactly the PER_CHANNEL scale index, and it is layout-independent,
// which is why the (kv_num_heads, 1, head_size) scale shape can be reused verbatim from GQA.

constexpr int kPagedInt8Min = -128;
constexpr int kPagedInt8Max = 127;
constexpr float kPagedFp8E4M3Max = 448.0f;

// True when the cache element type stores a quantized value that must be scaled on read/write.
template <typename TCACHE>
struct IsQuantizedCache : std::false_type {};
template <>
struct IsQuantizedCache<int8_t> : std::true_type {};
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
template <>
struct IsQuantizedCache<Float8E4M3FN> : std::true_type {};
#endif

// PER_CHANNEL scales are indexed by (kv_head * head_size + channel); PER_TENSOR uses scale[0].
// `channel_index` is that flattened kv-hidden offset.
__device__ __forceinline__ float GetCacheScale(const float* __restrict__ scale, const int channel_index,
                                               const bool per_channel) {
  if (scale == nullptr) {
    return 1.0f;
  }
  return per_channel ? scale[channel_index] : scale[0];
}

template <typename T, typename TCACHE>
__device__ __forceinline__ TCACHE QuantizeToCache(const T value, const float scale) {
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    const float inv_scale = (scale == 0.0f) ? 0.0f : (1.0f / scale);
    const int32_t q = static_cast<int32_t>(rintf(static_cast<float>(value) * inv_scale));
    return static_cast<int8_t>(max(kPagedInt8Min, min(kPagedInt8Max, q)));
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  } else if constexpr (std::is_same<TCACHE, Float8E4M3FN>::value) {
    const float inv_scale = (scale == 0.0f) ? 0.0f : (1.0f / scale);
    const float v = static_cast<float>(value) * inv_scale;
    return Float8E4M3FN(fmaxf(-kPagedFp8E4M3Max, fminf(kPagedFp8E4M3Max, v)));
#endif
  } else {
    return static_cast<TCACHE>(value);
  }
}

template <typename T, typename TCACHE>
__device__ __forceinline__ T DequantizeFromCache(const TCACHE value, const float scale) {
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    return static_cast<T>(static_cast<float>(value) * scale);
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  } else if constexpr (std::is_same<TCACHE, Float8E4M3FN>::value) {
    return static_cast<T>(value.ToFloat() * scale);
#endif
  } else {
    return static_cast<T>(value);
  }
}

////////// Auxiliary Kernels

template <typename T>
__global__ void UnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                    const int kv_num_heads, const int head_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= token_count * (num_heads + 2 * kv_num_heads) * head_size) {
    return;
  }
  const int q_hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int in_seq_stride = q_hidden_size + 2 * kv_hidden_size;

  int packed_i;
  if (tid < token_count * q_hidden_size) {
    const int token_id = tid / q_hidden_size;
    const int offset = tid % q_hidden_size;
    packed_i = token_id * in_seq_stride + offset;
  } else if (tid < token_count * (q_hidden_size + kv_hidden_size)) {
    const int id = tid - token_count * q_hidden_size;
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + offset;
  } else if (tid < token_count * (q_hidden_size + 2 * kv_hidden_size)) {
    const int id = tid - token_count * (q_hidden_size + kv_hidden_size);
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + kv_hidden_size + offset;
  }
  unpacked_qkv[tid] = packed_qkv[packed_i];
}

// Since QKV is unpacked into a single workspace buffer, this is similar to a transpose
template <typename T>
Status LaunchUnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                 const int kv_num_heads, const int head_size, cudaStream_t stream,
                                 const int max_threads_per_block) {
  const int threads = max_threads_per_block;
  const int blocks = (token_count * (num_heads + 2 * kv_num_heads) * head_size + threads - 1) / threads;
  UnpackQKVCumulative<T><<<blocks, threads, 0, stream>>>(packed_qkv, unpacked_qkv, token_count, num_heads, kv_num_heads,
                                                         head_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void UnpackV(const T* input, T* output, const int token_count, const int hidden_size,
                        const int packed_seq_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < token_count * hidden_size) {
    int offset = tid % hidden_size;
    int token_id = tid / hidden_size;
    int packed_i = token_id * packed_seq_stride + offset;
    output[tid] = input[packed_i];
  }
}

template <typename T>
Status LaunchUnpackCumulative(const T* input, T* output, const int token_count, const int hidden_size,
                              const int packed_seq_stride, cudaStream_t stream, const int max_threads_per_block) {
  const int threads = std::min(max_threads_per_block, token_count * hidden_size);
  const int blocks = (token_count * hidden_size + threads - 1) / threads;
  UnpackV<T><<<blocks, threads, 0, stream>>>(input, output, token_count, hidden_size, packed_seq_stride);
  return CUDA_CALL(cudaGetLastError());
}

// Fused per-head RMSNorm (QK-Norm) + rotary embedding over the unpadded TxNxH layout.
//
// Both transformations are pure prologue work on Q and K, so fusing them keeps a single read of the
// (possibly packed) input and a single write of the workspace, and it keeps the whole prologue
// independent of which attention backend runs afterwards. Order matters: QK-Norm is applied to the
// raw projection *before* RoPE, matching the reference implementations (and GroupQueryAttention's
// UnpackRoPEAppend), so the value that lands in the paged KV cache is normalized-then-rotated K.
//
// Set norm_weight == nullptr to skip normalization, and rotary_embedding_dim == 0 to skip rotary
// (the kernel then degenerates to an unpack/copy). At least one of the two must be enabled,
// otherwise the caller should not launch this kernel at all.
//
// blockDim.x is the smallest power of two >= head_size so the reduction tree is exact; threads with
// h >= head_size participate in the reduction (contributing 0) but perform no global access.
template <typename T>
__global__ void QkNormRotaryTNH(T* output,                            // TxNxH
                                const T* input,                       // TxNxH
                                const T* cos_cache,                   // Mx(H/2)
                                const T* sin_cache,                   // Mx(H/2)
                                const int32_t* past_seqlens,          // B
                                const int32_t* cumulative_seqlens_q,  // B+1
                                const T* norm_weight,                 // H, or nullptr
                                const float epsilon,
                                const int head_size,
                                const int rotary_embedding_dim,
                                const bool interleaved,
                                const int3 in_strides,     // TxNxH
                                const int3 out_strides) {  // TxNxH
  // Use .x in innermost loop to access global memory efficiently

  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;
  const int h = threadIdx.x;

  const int sequence_length = cumulative_seqlens_q[b + 1] - cumulative_seqlens_q[b];
  // Uniform across the block, so returning here cannot desynchronize the barriers below.
  if (s >= sequence_length) {
    return;
  }

  // Layout: blockDim.x floats for the reduction tree, then head_size elements of T holding the
  // (optionally normalized) head so the rotary step can read its partner lane without a second
  // global load. The float array comes first to keep both regions naturally aligned.
  extern __shared__ char smem[];
  float* reduce_buffer = reinterpret_cast<float*>(smem);
  T* head_values = reinterpret_cast<T*>(reduce_buffer + blockDim.x);

  const int t = cumulative_seqlens_q[b] + s;  // t is the index of the token in the unpadded input/output
  const T* input_data = input + t * in_strides.x + n * in_strides.y;
  T* output_data = output + t * out_strides.x + n * out_strides.y;

  const bool valid = h < head_size;
  float value = valid ? static_cast<float>(input_data[h]) : 0.0f;

  if (norm_weight != nullptr) {
    // RMSNorm across head_size, accumulated in fp32 regardless of T.
    reduce_buffer[h] = value * value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (h < static_cast<int>(stride)) {
        reduce_buffer[h] += reduce_buffer[h + stride];
      }
      __syncthreads();
    }
    const float inv_rms = rsqrtf(reduce_buffer[0] / static_cast<float>(head_size) + epsilon);
    if (valid) {
      value = value * inv_rms * static_cast<float>(norm_weight[h]);
    }
  }

  if (valid) {
    head_values[h] = static_cast<T>(value);
  }
  __syncthreads();

  if (!valid) {
    return;
  }

  // rotary_embedding_dim == 0 makes every lane take this branch, which is the pure
  // normalize-and-copy (or plain unpack) path. past_seqlens / cos_cache / sin_cache are then
  // never dereferenced and may be null.
  if (h >= rotary_embedding_dim) {
    output_data[h] = head_values[h];
    return;
  }

  // Cache is (M, H/2)
  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
  const int position_id = past_seqlens[b] + s;
  const int cache_offset = position_id * half_rotary_embedding_dim;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (h / 2) % half_rotary_embedding_dim;
    sign = (h % 2 == 0) ? -1 : 1;
    j = (h % 2 == 0) ? h + 1 : h - 1;  // i - sign
  } else {
    cache_idx = h % half_rotary_embedding_dim;
    sign = (h < half_rotary_embedding_dim) ? -1 : 1;
    j = (h + half_rotary_embedding_dim) % rotary_embedding_dim;
  }
  output_data[h] = head_values[h] * cos_data[cache_idx] + sign * head_values[j] * sin_data[cache_idx];
}

// Launches the fused QK-Norm / rotary prologue. Pass norm_weight == nullptr to disable QK-Norm and
// rotary_embedding_dim == 0 to disable rotary; the caller is responsible for not invoking this when
// both are disabled and the input is not packed.
template <typename T>
Status LaunchQkNormRotaryKernel(cudaStream_t stream, T* output, const T* input, const int32_t* past_seqlens,
                                const int32_t* cumulative_seqlens_q, const T* cos_cache, const T* sin_cache,
                                const T* norm_weight, const float epsilon, const int batch_size,
                                const int max_seqlen_q, const int num_heads, const int head_size,
                                const int rotary_embedding_dim, const bool interleaved, const int in_seq_stride,
                                const int max_threads_per_block) {
  if (batch_size == 0 || max_seqlen_q == 0 || num_heads == 0) {
    return Status::OK();
  }
  int3 in_strides = {in_seq_stride <= 0 ? num_heads * head_size : in_seq_stride, head_size, 1};
  int3 out_strides = {num_heads * head_size, head_size, 1};
  // Round up to a power of two so the reduction tree halves exactly.
  int tpb = 32;
  while (tpb < head_size) {
    tpb <<= 1;
  }
  ORT_ENFORCE(tpb <= max_threads_per_block,
              "PagedAttention prologue requires head_size rounded up to a power of two (", tpb,
              ") to be <= max_threads_per_block (", max_threads_per_block, ")");

  const size_t shared_bytes = static_cast<size_t>(tpb) * sizeof(float) + static_cast<size_t>(head_size) * sizeof(T);
  const dim3 grid(max_seqlen_q, batch_size, num_heads);
  const dim3 block(tpb);
  QkNormRotaryTNH<<<grid, block, shared_bytes, stream>>>(
      output, input, cos_cache, sin_cache, past_seqlens, cumulative_seqlens_q, norm_weight, epsilon, head_size,
      rotary_embedding_dim, interleaved, in_strides, out_strides);
  return CUDA_CALL(cudaGetLastError());
}

// Single-block inclusive scan over the per-sequence KV lengths. One block loops over the batch in
// kBlockSize-sized tiles carrying a running total, so there is no cap on batch_size (the previous
// implementation launched independent blocks whose cub::BlockScan did not compose, which silently
// produced wrong offsets past 256 concurrent sequences).
template <int kBlockSize>
__global__ void GetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                       const int32_t* past_seqlens, const int batch_size) {
  typedef cub::BlockScan<int, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ int running_total;

  if (threadIdx.x == 0) {
    cumulative_seqlens_kv[0] = 0;
    running_total = 0;
  }
  __syncthreads();

  for (int base = 0; base < batch_size; base += kBlockSize) {
    const int id = base + static_cast<int>(threadIdx.x);
    // Sum past_seqlens to the new sequence length (which we get by subtracting cumulative_seqlens_q),
    // then inclusive-scan across present sequence lengths.
    const int length = (id < batch_size)
                           ? past_seqlens[id] + cumulative_seqlens_q[id + 1] - cumulative_seqlens_q[id]
                           : 0;
    int prefix = 0;
    int aggregate = 0;
    BlockScan(temp_storage).InclusiveSum(length, prefix, aggregate);
    if (id < batch_size) {
      cumulative_seqlens_kv[id + 1] = running_total + prefix;
    }
    __syncthreads();  // all reads of running_total and of temp_storage are done
    if (threadIdx.x == 0) {
      running_total += aggregate;
    }
    __syncthreads();  // running_total visible, temp_storage safe to reuse
  }
}

Status LaunchGetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                    const int32_t* past_seqlens, const int batch_size, cudaStream_t stream) {
  constexpr int kThreads = 256;
  GetCumulativeSeqlensKV<kThreads><<<1, kThreads, 0, stream>>>(cumulative_seqlens_kv, cumulative_seqlens_q,
                                                               past_seqlens, batch_size);
  return CUDA_CALL(cudaGetLastError());
}

// Resolves the flat cache slot that a query token's K/V is written to, in the cache viewed as
// [num_blocks * block_size, kv_num_heads, head_size]. A negative result suppresses the store.
//
// DerivedSlotResolver reproduces the legacy behavior: append the token at
// past_seqlens[b] + (token_id - cumulative_seqlens_q[b]) of its own sequence. The binary search is
// guarded against token_id >= cumulative_seqlens_q[batch_size], which previously walked off the end
// of past_seqlens / block_table.
struct DerivedSlotResolver {
  const int* __restrict__ block_table;
  const int* __restrict__ past_seqlens;
  const int* __restrict__ cumulative_seqlens_q;
  int batch_size;
  int max_num_blocks_per_seq;
  int block_size;

  __device__ __forceinline__ int operator()(int token_id) const {
    if (token_id < 0 || token_id >= cumulative_seqlens_q[batch_size]) {
      return -1;
    }
    // cumulative_seqlens_q is a non-decreasing prefix sum, so binary search finds the owning
    // sequence in log2(batch_size) steps instead of the previous O(batch_size) scan.
    int left = 0;
    int right = batch_size - 1;
    while (left < right) {
      const int mid = left + (right - left) / 2;
      if (token_id < cumulative_seqlens_q[mid + 1]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const int batch_id = left;
    const int position = past_seqlens[batch_id] + (token_id - cumulative_seqlens_q[batch_id]);
    const int block_idx_in_seq = position / block_size;
    if (block_idx_in_seq >= max_num_blocks_per_seq) {
      return -1;
    }
    const int block_id = block_table[batch_id * max_num_blocks_per_seq + block_idx_in_seq];
    if (block_id < 0) {  // unmapped block
      return -1;
    }
    return block_id * block_size + position % block_size;
  }
};

// ExplicitSlotResolver consumes the scheduler-provided slot_mapping (input 10) directly. This is
// what prefix caching, chunked prefill and speculative decoding need: the scheduler, not the
// kernel, owns block placement. It also removes the per-thread binary search entirely.
struct ExplicitSlotResolver {
  const int* __restrict__ slot_mapping;

  __device__ __forceinline__ int operator()(int token_id) const {
    return slot_mapping[token_id];
  }
};

template <typename T, typename TCACHE, typename SlotResolver>
__global__ void ReshapeAndCache(const T* __restrict__ key, const T* __restrict__ value,
                                TCACHE* __restrict__ key_cache, TCACHE* __restrict__ value_cache,
                                const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                                const bool k_per_channel, const bool v_per_channel,
                                const SlotResolver resolver, const int64_t total_elems,
                                const int kv_hidden_size, const int key_stride, const int value_stride,
                                const int64_t num_slots) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
       tid < total_elems;
       tid += stride) {
    const int token_id = static_cast<int>(tid / kv_hidden_size);
    const int hidden_offset = static_cast<int>(tid % kv_hidden_size);

    const int slot = resolver(token_id);
    // slot < 0 means "do not write this token" (unmapped block, or an explicit -1 in slot_mapping
    // for a prefix-cache hit / rejected speculative token). The Q of such a token still attends.
    if (slot < 0 || slot >= num_slots) {
      continue;
    }

    const int64_t key_id = static_cast<int64_t>(token_id) * key_stride + hidden_offset;
    const int64_t value_id = static_cast<int64_t>(token_id) * value_stride + hidden_offset;
    const int64_t dst_id = static_cast<int64_t>(slot) * kv_hidden_size + hidden_offset;
    // hidden_offset is (kv_head * head_size + channel), which is exactly the PER_CHANNEL scale
    // index. For an unquantized cache the scale pointers are null and this compiles to a copy.
    key_cache[dst_id] =
        QuantizeToCache<T, TCACHE>(key[key_id], GetCacheScale(k_scale, hidden_offset, k_per_channel));
    value_cache[dst_id] =
        QuantizeToCache<T, TCACHE>(value[value_id], GetCacheScale(v_scale, hidden_offset, v_per_channel));
  }
}

template <typename T, typename TCACHE, typename SlotResolver>
Status LaunchReshapeAndCacheImpl(const T* key, const T* value, TCACHE* key_cache, TCACHE* value_cache,
                                 const float* k_scale, const float* v_scale, const bool k_per_channel,
                                 const bool v_per_channel, const SlotResolver& resolver, const int token_count,
                                 const int kv_hidden_size, const int key_stride, const int value_stride,
                                 const int64_t num_slots, cudaStream_t stream, const int max_threads_per_block) {
  const int64_t total_elems = static_cast<int64_t>(token_count) * kv_hidden_size;
  if (total_elems == 0) {
    return Status::OK();
  }
  const int threads = static_cast<int>(std::min<int64_t>(max_threads_per_block, total_elems));
  const int blocks = static_cast<int>(std::min<int64_t>((total_elems + threads - 1) / threads, 65535));
  ReshapeAndCache<T, TCACHE, SlotResolver><<<blocks, threads, 0, stream>>>(
      key, value, key_cache, value_cache, k_scale, v_scale, k_per_channel, v_per_channel, resolver, total_elems,
      kv_hidden_size, key_stride, value_stride, num_slots);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename TCACHE>
Status LaunchReshapeAndCache(const T* key, const T* value, TCACHE* key_cache, TCACHE* value_cache,
                             const float* k_scale, const float* v_scale, const bool k_per_channel,
                             const bool v_per_channel, const int* block_table,
                             const int* past_seqlens, const int* cumulative_seqlens_q, const int* slot_mapping,
                             const int batch_size, const int max_num_blocks_per_seq, const int token_count,
                             const int kv_hidden_size, const int block_size, const int num_blocks,
                             const int key_stride, const int value_stride, cudaStream_t stream,
                             const int max_threads_per_block) {
  const int64_t num_slots = static_cast<int64_t>(num_blocks) * block_size;
  if (slot_mapping != nullptr) {
    ExplicitSlotResolver resolver{slot_mapping};
    return LaunchReshapeAndCacheImpl<T, TCACHE, ExplicitSlotResolver>(
        key, value, key_cache, value_cache, k_scale, v_scale, k_per_channel, v_per_channel, resolver,
        token_count, kv_hidden_size, key_stride, value_stride, num_slots, stream, max_threads_per_block);
  }
  DerivedSlotResolver resolver{block_table, past_seqlens, cumulative_seqlens_q, batch_size,
                               max_num_blocks_per_seq, block_size};
  return LaunchReshapeAndCacheImpl<T, TCACHE, DerivedSlotResolver>(
      key, value, key_cache, value_cache, k_scale, v_scale, k_per_channel, v_per_channel, resolver,
      token_count, kv_hidden_size, key_stride, value_stride, num_slots, stream, max_threads_per_block);
}

// Exact attention-sink epilogue. FlashAttention returns
//   lse[t,h] = log(sum_j exp(x_j))  and  o[t,h] = (sum_j exp(x_j) v_j) / sum_j exp(x_j),
// and a sink only adds the extra logit s_h to the denominator, so the corrected output is the
// elementwise rescale  o *= exp(lse) / (exp(lse) + exp(s_h)) = 1 / (1 + exp(s_h - lse)).
// This is numerically stable for both signs of (s_h - lse), needs no change to the Flash kernel,
// and composes with sliding window, softcap and GQA grouping because lse already reflects the mask.
// smooth_softmax without a head_sink input is the same formula with s_h = 0.
template <typename T>
__global__ void ApplyHeadSink(T* __restrict__ output, const float* __restrict__ softmax_lse,
                              const T* __restrict__ head_sink, const int token_count, const int num_heads,
                              const int head_size, const int64_t total_elems) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t num_heads_times_head = static_cast<int64_t>(num_heads) * head_size;
  for (int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
       tid < total_elems;
       tid += stride) {
    const int head_id = static_cast<int>((tid / head_size) % num_heads);
    const int token_id = static_cast<int>(tid / num_heads_times_head);
    // Varlen LSE layout is [num_heads, token_count] (Flash's unpadded_lse with num_splits <= 1).
    const float lse = softmax_lse[static_cast<int64_t>(head_id) * token_count + token_id];
    const float sink = (head_sink == nullptr) ? 0.0f : static_cast<float>(head_sink[head_id]);
    // lse == +inf marks a fully masked row (output already zero) -> factor 1; lse == -inf gives
    // factor 0. expf saturates correctly in both cases, so no special casing is needed.
    const float factor = 1.0f / (1.0f + expf(sink - lse));
    output[tid] = static_cast<T>(static_cast<float>(output[tid]) * factor);
  }
}

template <typename T>
Status LaunchApplyHeadSink(T* output, const float* softmax_lse, const T* head_sink, const int token_count,
                           const int num_heads, const int head_size, cudaStream_t stream,
                           const int max_threads_per_block) {
  const int64_t total_elems = static_cast<int64_t>(token_count) * num_heads * head_size;
  if (total_elems == 0) {
    return Status::OK();
  }
  const int threads = static_cast<int>(std::min<int64_t>(max_threads_per_block, total_elems));
  const int blocks = static_cast<int>(std::min<int64_t>((total_elems + threads - 1) / threads, 65535));
  ApplyHeadSink<T><<<blocks, threads, 0, stream>>>(output, softmax_lse, head_sink, token_count, num_heads,
                                                   head_size, total_elems);
  return CUDA_CALL(cudaGetLastError());
}

// Gather paged KV into packed-varlen [total_kv_tokens, out_num_heads, head_size], dequantizing on
// the fly when the cache is quantized. out_num_heads == num_heads expands GQA groups (what the
// CUTLASS memory-efficient kernel needs); out_num_heads == kv_num_heads keeps the grouped layout
// (what FlashAttention's non-paged varlen entry point needs).
// total_elems = total_kv_tokens * out_num_heads * head_size can exceed INT32_MAX for realistic
// large-context GQA configs (e.g., 2M tokens * 64 * 128 = 16.4B), so the linear index is int64_t
// and the kernel uses a grid-stride loop instead of a single (tid >= total_elems) early-exit.
template <typename T, typename TCACHE>
__global__ void GatherAndExpandPagedKVCache(const TCACHE* __restrict__ key_cache,
                                            const TCACHE* __restrict__ value_cache,
                                            T* __restrict__ gathered_key,
                                            T* __restrict__ gathered_value,
                                            const float* __restrict__ k_scale,
                                            const float* __restrict__ v_scale,
                                            const bool k_per_channel,
                                            const bool v_per_channel,
                                            const int* __restrict__ block_table,
                                            const int* __restrict__ cumulative_seqlens_kv,
                                            const int batch_size,
                                            const int num_heads,
                                            const int kv_num_heads,
                                            const int head_size,
                                            const int block_size,
                                            const int max_num_blocks_per_seq,
                                            const int64_t total_elems) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t num_heads_times_head = static_cast<int64_t>(num_heads) * head_size;
  const int q_kv_head_ratio = num_heads / kv_num_heads;
  const int64_t page_stride = static_cast<int64_t>(block_size) * kv_num_heads * head_size;

  for (int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
       tid < total_elems;
       tid += stride) {
    const int h = static_cast<int>(tid % head_size);
    const int head_id = static_cast<int>((tid / head_size) % num_heads);
    const int token_id = static_cast<int>(tid / num_heads_times_head);

    // cumulative_seqlens_kv is a prefix sum of non-negative per-batch KV lengths
    // (past_seqlens[i] + new_tokens[i]), so it is monotonically non-decreasing for
    // any valid op input — the same assumption the previous linear scan made.
    // Binary-search for the batch this token belongs to: log2(batch_size) is strictly
    // better than the linear scan, which ran once per (token, head, h) element and
    // multiplied its cost by num_heads * head_size.
    int left = 0;
    int right = batch_size;
    while (left < right) {
      const int mid = left + (right - left) / 2;
      if (token_id < cumulative_seqlens_kv[mid + 1]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const int batch_id = left;

    // Defensive: a malformed cumulative_seqlens_kv (or a total_kv_tokens that disagrees with it)
    // would leave batch_id == batch_size and walk off the end of block_table.
    if (batch_id >= batch_size) {
      continue;
    }

    const int pos = token_id - cumulative_seqlens_kv[batch_id];
    const int block_idx_in_seq = pos / block_size;
    const int block_offset = pos % block_size;
    if (block_idx_in_seq >= max_num_blocks_per_seq) {
      continue;
    }
    const int block_id = block_table[batch_id * max_num_blocks_per_seq + block_idx_in_seq];
    if (block_id < 0) {
      gathered_key[tid] = static_cast<T>(0.f);
      gathered_value[tid] = static_cast<T>(0.f);
      continue;
    }

    // GQA expansion: each output head maps to kv_head_id = head_id / (num_heads / kv_num_heads).
    // For MHA (num_heads == kv_num_heads) this is the identity.
    const int kv_head_id = head_id / q_kv_head_ratio;
    const int channel_index = kv_head_id * head_size + h;

    const int64_t paged_idx = static_cast<int64_t>(block_id) * page_stride +
                              static_cast<int64_t>(block_offset) * kv_num_heads * head_size +
                              kv_head_id * head_size +
                              h;

    gathered_key[tid] =
        DequantizeFromCache<T, TCACHE>(key_cache[paged_idx], GetCacheScale(k_scale, channel_index, k_per_channel));
    gathered_value[tid] =
        DequantizeFromCache<T, TCACHE>(value_cache[paged_idx], GetCacheScale(v_scale, channel_index, v_per_channel));
  }
}

template <typename T, typename TCACHE>
Status LaunchGatherAndExpandPagedKVCache(const TCACHE* key_cache, const TCACHE* value_cache,
                                         T* gathered_key, T* gathered_value,
                                         const float* k_scale, const float* v_scale,
                                         const bool k_per_channel, const bool v_per_channel,
                                         const int* block_table, const int* cumulative_seqlens_kv,
                                         const int batch_size, const int num_heads,
                                         const int kv_num_heads, const int head_size,
                                         const int block_size, const int max_num_blocks_per_seq,
                                         const int total_kv_tokens, cudaStream_t stream,
                                         const int max_threads_per_block) {
  const int64_t total_elems = static_cast<int64_t>(total_kv_tokens) * num_heads * head_size;
  if (total_elems == 0) {
    return Status::OK();
  }
  // The kernel uses a grid-stride loop, so the block count is clamped rather than allowed to
  // overflow int for very large contexts.
  const int threads = static_cast<int>(std::min<int64_t>(max_threads_per_block, total_elems));
  const int blocks = static_cast<int>(std::min<int64_t>((total_elems + threads - 1) / threads, 65535));
  GatherAndExpandPagedKVCache<T, TCACHE><<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, gathered_key, gathered_value, k_scale, v_scale, k_per_channel, v_per_channel,
      block_table, cumulative_seqlens_kv,
      batch_size, num_heads, kv_num_heads, head_size,
      block_size, max_num_blocks_per_seq, total_elems);
  return CUDA_CALL(cudaGetLastError());
}

////////// Launch Kernels

#if USE_FLASH_ATTENTION
template <typename T, typename TCACHE>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    float scale) {
  // Get parameters
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const float softcap = parameters.softcap;
  bool is_bf16 = std::is_same<T, BFloat16>::value;
  const int local_window_size = parameters.local_window_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int block_size = parameters.block_size;
  // Host-computed actual max from paged_attention.cc. Used as both
  // `params.seqlen_q` for mha_varlen_fwd and grid.x for the rotary kernel.
  const int max_query_len = data.max_query_len;

  T* query = const_cast<T*>(data.query);
  T* key;
  T* value;
  if (!parameters.is_packed_qkv) {
    key = const_cast<T*>(data.key);
    value = const_cast<T*>(data.value);
  } else {
    key = reinterpret_cast<T*>(query) + static_cast<size_t>(num_heads * head_size);
    value = reinterpret_cast<T*>(key) + static_cast<size_t>(kv_num_heads * head_size);
  }

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across FA and MEA dispatch paths so the host can also read total_kv_tokens.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;

  if (parameters.do_rotary || parameters.use_qk_norm) {
    // Fused QK-Norm + rotary prologue. Also unpacks Q and K in case of packed_qkv.
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    const int rotary_dim = parameters.do_rotary ? parameters.rotary_dim : 0;
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.q_norm_weight, parameters.qk_norm_epsilon, batch_size, max_query_len, num_heads, head_size,
        rotary_dim, parameters.rotary_interleaved, packed_seq_stride, max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.k_norm_weight, parameters.qk_norm_epsilon, batch_size, max_query_len, kv_num_heads, head_size,
        rotary_dim, parameters.rotary_interleaved, packed_seq_stride, max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
  } else if (parameters.is_packed_qkv) {
    // Only unpack Q. K and V are unpacked by ReshapeAndCache.
    auto q_buffer = data.workspace_buffer;
    const int packed_seq_stride = q_hidden_size + 2 * kv_hidden_size;
    ORT_RETURN_IF_ERROR(LaunchUnpackCumulative<T>(
        query, q_buffer, token_count, q_hidden_size, packed_seq_stride, stream, max_threads_per_block));
    query = q_buffer;
  }

  // Insert key and value into block-based KV cache. The prologue (if it ran) already densified K
  // into the workspace, so only the "no prologue" packed-QKV case still needs the packed stride.
  const bool k_is_packed = parameters.is_packed_qkv && !(parameters.do_rotary || parameters.use_qk_norm);
  int* block_table = const_cast<int*>(data.block_table);
  const int key_stride = k_is_packed ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;
  ORT_RETURN_IF_ERROR((LaunchReshapeAndCache<T, TCACHE>(
      key, value, data.key_cache, data.value_cache, data.k_scale, data.v_scale, k_per_channel, v_per_channel,
      block_table, past_seqlens, cumulative_seqlens_q, data.slot_mapping, batch_size, max_num_blocks_per_seq,
      token_count, kv_hidden_size, block_size, parameters.num_blocks, key_stride, value_stride, stream,
      max_threads_per_block)));

  // Launch kernel
  void* q = reinterpret_cast<void*>(query);
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);

  if constexpr (IsQuantizedCache<TCACHE>::value) {
    // FlashAttention cannot read a quantized page, so dequantize the live context into a dense
    // packed-varlen [total_kv_tokens, kv_num_heads, head_size] buffer (no GQA expansion — Flash
    // does the grouping itself) and use the non-paged varlen entry point. That path leaves
    // params.num_splits at 0 exactly like the paged one, so the fp32 [num_heads, token_count]
    // softmax_lse layout the head-sink epilogue relies on is unchanged.
    ORT_RETURN_IF_ERROR((LaunchGatherAndExpandPagedKVCache<T, TCACHE>(
        data.key_cache, data.value_cache, data.gathered_key, data.gathered_value,
        data.k_scale, data.v_scale, k_per_channel, v_per_channel,
        block_table, cumulative_seqlens_kv, batch_size, /*num_heads*/ kv_num_heads, kv_num_heads,
        head_size, block_size, max_num_blocks_per_seq, data.total_kv_tokens, stream, max_threads_per_block)));

    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
        device_prop, stream, q, reinterpret_cast<void*>(data.gathered_key),
        reinterpret_cast<void*>(data.gathered_value), output, cumulative_seqlens_q, cumulative_seqlens_kv,
        /*seqused_k*/ nullptr, /*block_table*/ nullptr, softmax_lse, batch_size, num_heads, kv_num_heads, head_size,
        max_query_len, data.max_kv_len, token_count, scale, softcap, /*is_causal*/ true, is_bf16,
        local_window_size - 1));
  } else {
    void* key_cache = reinterpret_cast<void*>(data.key_cache);
    void* value_cache = reinterpret_cast<void*>(data.value_cache);
    const int max_seq_len = max_num_blocks_per_seq * block_size;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
        device_prop, stream, q, key_cache, value_cache, output, cumulative_seqlens_q, cumulative_seqlens_kv,
        /*seqused_k*/ nullptr, block_table, softmax_lse, batch_size, num_heads, kv_num_heads, head_size,
        max_query_len, max_seq_len, token_count, scale, softcap, /*is_causal*/ true, is_bf16, local_window_size - 1,
        max_num_blocks_per_seq, block_size));
  }

  if (parameters.use_smooth_softmax) {
    // Rescale by the softmax denominator that the sink logit adds. mha_varlen_fwd leaves
    // params.num_splits at 0, so the split-combine kernel never runs and softmax_lse carries the
    // unpadded [num_heads, token_count] fp32 layout this epilogue expects. If varlen ever enables
    // num_splits > 1, both the layout and this epilogue must be revisited.
    ORT_RETURN_IF_ERROR(LaunchApplyHeadSink<T>(data.output, data.softmax_lse, data.head_sink, token_count,
                                               num_heads, head_size, stream, max_threads_per_block));
  }

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("flash attention output", data.output, token_count, num_heads, head_size);

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
// Fallback when FlashAttention is unavailable (SM<80 or ORT_DISABLE_FLASH_ATTENTION=1).
// Mirrors the FlashAttention preprocessing (rotary, unpack, ReshapeAndCache), then gathers
// the paged KV cache into a packed-varlen [total_kv_tokens, num_heads, head_size] buffer and
// dispatches to CUTLASS memory-efficient attention via its seqstart_q / seqstart_k varlen ABI.
// Caller must populate data.gathered_key / data.gathered_value / data.total_kv_tokens.
template <typename T, typename TCACHE>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const int block_size = parameters.block_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int local_window_size = parameters.local_window_size;
  const int total_kv_tokens = data.total_kv_tokens;
  // Use the caller-computed actual max of per-batch new-query lengths, not the
  // `token_count - batch_size + 1` heuristic: the heuristic assumes >=1 new token per batch
  // and underestimates otherwise, which would silently drop query tokens from the
  // rotary grid and from MEA's `grid_x = ceil_div(sequence_length, kQueriesPerBlock)`.
  const int max_query_len = data.max_query_len;

  T* query = const_cast<T*>(data.query);
  T* key;
  T* value;
  if (!parameters.is_packed_qkv) {
    key = const_cast<T*>(data.key);
    value = const_cast<T*>(data.value);
  } else {
    key = reinterpret_cast<T*>(query) + static_cast<size_t>(num_heads * head_size);
    value = reinterpret_cast<T*>(key) + static_cast<size_t>(kv_num_heads * head_size);
  }

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across FA and MEA dispatch paths.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;

  if (parameters.do_rotary || parameters.use_qk_norm) {
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    const int rotary_dim = parameters.do_rotary ? parameters.rotary_dim : 0;
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.q_norm_weight, parameters.qk_norm_epsilon, batch_size, max_query_len, num_heads, head_size,
        rotary_dim, parameters.rotary_interleaved, packed_seq_stride, max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.k_norm_weight, parameters.qk_norm_epsilon, batch_size, max_query_len, kv_num_heads, head_size,
        rotary_dim, parameters.rotary_interleaved, packed_seq_stride, max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
  } else if (parameters.is_packed_qkv) {
    auto q_buffer = data.workspace_buffer;
    const int packed_seq_stride = q_hidden_size + 2 * kv_hidden_size;
    ORT_RETURN_IF_ERROR(LaunchUnpackCumulative<T>(
        query, q_buffer, token_count, q_hidden_size, packed_seq_stride, stream, max_threads_per_block));
    query = q_buffer;
  }

  const bool k_is_packed = parameters.is_packed_qkv && !(parameters.do_rotary || parameters.use_qk_norm);
  int* block_table = const_cast<int*>(data.block_table);
  const int key_stride = k_is_packed ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;
  ORT_RETURN_IF_ERROR((LaunchReshapeAndCache<T, TCACHE>(
      key, value, data.key_cache, data.value_cache, data.k_scale, data.v_scale, k_per_channel, v_per_channel,
      block_table, past_seqlens, cumulative_seqlens_q, data.slot_mapping, batch_size, max_num_blocks_per_seq,
      token_count, kv_hidden_size, block_size, parameters.num_blocks, key_stride, value_stride, stream,
      max_threads_per_block)));

  ORT_RETURN_IF_ERROR((LaunchGatherAndExpandPagedKVCache<T, TCACHE>(
      data.key_cache, data.value_cache, data.gathered_key, data.gathered_value,
      data.k_scale, data.v_scale, k_per_channel, v_per_channel,
      block_table, cumulative_seqlens_kv, batch_size, num_heads, kv_num_heads,
      head_size, block_size, max_num_blocks_per_seq, total_kv_tokens, stream, max_threads_per_block)));

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_bf16 = std::is_same<T, BFloat16>::value;
  p.is_half = !p.is_bf16 && (sizeof(T) == 2);
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = max_query_len;
  p.kv_sequence_length = total_kv_tokens;
  p.max_sequence_length = total_kv_tokens;
  p.qk_head_size = head_size;
  p.v_head_size = head_size;
  p.causal = true;
  p.scale = scale;
  p.softcap = parameters.softcap;
  p.local_window_size = local_window_size;
  p.seqstart_q_ptr = cumulative_seqlens_q;
  p.seqstart_k_ptr = cumulative_seqlens_kv;
  p.seqlen_k_ptr = nullptr;
  p.query = query;
  p.key = data.gathered_key;
  p.value = data.gathered_value;
  p.attn_bias = nullptr;
  p.is_kv_bsnh = true;
  p.has_custom_right_padding = false;
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(head_size, sizeof(T) == sizeof(float))
                    ? data.fmha_buffer
                    : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("mea paged attention output", data.output, token_count, num_heads, head_size);

  return Status::OK();
}
#endif

////////// API Functions

template <typename T, typename TCACHE>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& /*cublas*/,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data, scale);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return EfficientAttention(device_prop, stream, parameters, data, scale);
  }
#endif

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No PagedAttention kernel available for the current configuration.");
}

#define INSTANTIATE_PAGED_ATTENTION(T, TCACHE)       \
  template struct PagedAttentionData<T, TCACHE>;     \
  template Status QkvToContext<T, TCACHE>(           \
      const cudaDeviceProp& device_prop,             \
      cublasHandle_t& cublas,                        \
      Stream* ort_stream,                            \
      contrib::PagedAttentionParameters& parameters, \
      PagedAttentionData<T, TCACHE>& data);

INSTANTIATE_PAGED_ATTENTION(half, half)
INSTANTIATE_PAGED_ATTENTION(BFloat16, BFloat16)
INSTANTIATE_PAGED_ATTENTION(half, int8_t)
INSTANTIATE_PAGED_ATTENTION(BFloat16, int8_t)
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
INSTANTIATE_PAGED_ATTENTION(half, Float8E4M3FN)
INSTANTIATE_PAGED_ATTENTION(BFloat16, Float8E4M3FN)
#endif

#undef INSTANTIATE_PAGED_ATTENTION

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
