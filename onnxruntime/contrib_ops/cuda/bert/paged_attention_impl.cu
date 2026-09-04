// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cfloat>  // FLT_MAX
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
#include "contrib_ops/cuda/bert/xqa/xqa_paged_loader.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include <cublas_v2.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__ void SanitizeBlockTableKernel(const int32_t* block_table, int32_t* sanitized_block_table,
                                         int element_count, int num_blocks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < element_count) {
    const int32_t block_id = block_table[index];
    sanitized_block_table[index] = block_id >= -1 && block_id < num_blocks ? block_id : 0;
  }
}

Status LaunchSanitizeBlockTable(const int32_t* block_table, int32_t* sanitized_block_table,
                                int element_count, int num_blocks, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  const int blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  SanitizeBlockTableKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      block_table, sanitized_block_table, element_count, num_blocks);
  return CUDA_CALL(cudaGetLastError());
}

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
// `rotary_offset` is the first channel of the head that RoPE covers: the rotated span is
// [rotary_offset, rotary_offset + rotary_embedding_dim) and every channel outside it is copied
// through. rotary_offset == 0 is the shipped prefix-RoPE behavior; absorbed MLA rotates only the
// k_pe suffix and passes rotary_offset == kv_lora_rank.
//
// blockDim.x is the smallest power of two >= head_size so the reduction tree is exact; threads with
// h >= head_size participate in the reduction (contributing 0) but perform no global access.
//
// The grid is indexed by *global token*, not by (sequence position, sequence): the owning sequence
// is recovered from cumulative_seqlens_q with a binary search. That keeps the launch exactly
// token_count * num_heads blocks for any raggedness -- there is no per-sequence padding to skip --
// and, more importantly, it removes the last dependence of the prologue on a host-computed
// max_query_len, which could only be obtained with a device-to-host synchronization
// (docs/contrib_ops/cuda/paged_attention.md section 4.7).
template <typename T>
__global__ void QkNormRotaryTNH(T* output,                            // TxNxH
                                const T* input,                       // TxNxH
                                const T* cos_cache,                   // Mx(H/2)
                                const T* sin_cache,                   // Mx(H/2)
                                const int32_t* past_seqlens,          // B
                                const int32_t* cumulative_seqlens_q,  // B+1
                                const T* norm_weight,                 // H, or nullptr
                                const float epsilon,
                                const int batch_size,
                                const int head_size,
                                const int rotary_embedding_dim,
                                const int rotary_offset,
                                const bool interleaved,
                                const int3 in_strides,     // TxNxH
                                const int3 out_strides) {  // TxNxH
  // Use .x in innermost loop to access global memory efficiently

  const int t = blockIdx.x;  // index of the token in the unpadded input/output
  const int n = blockIdx.y;
  const int h = threadIdx.x;

  // cumulative_seqlens_q is non-decreasing, so this finds the first sequence whose exclusive end
  // exceeds t -- which is the owning sequence, and correctly skips sequences with no new token.
  int left = 0;
  int right = batch_size;
  while (left < right) {
    const int mid = left + (right - left) / 2;
    if (t < cumulative_seqlens_q[mid + 1]) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  const int b = left;
  // Defensive: a malformed cumulative_seqlens_q whose total disagrees with token_count would
  // otherwise read past the end of past_seqlens. Uniform across the block.
  if (b >= batch_size) {
    return;
  }
  const int s = t - cumulative_seqlens_q[b];  // position of the token within its own sequence

  // Layout: blockDim.x floats for the reduction tree, then head_size elements of T holding the
  // (optionally normalized) head so the rotary step can read its partner lane without a second
  // global load. The float array comes first to keep both regions naturally aligned.
  extern __shared__ char smem[];
  float* reduce_buffer = reinterpret_cast<float*>(smem);
  T* head_values = reinterpret_cast<T*>(reduce_buffer + blockDim.x);

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

  // A channel outside [rotary_offset, rotary_offset + rotary_embedding_dim) is copied through.
  // rotary_embedding_dim == 0 makes every lane take this branch, which is the pure
  // normalize-and-copy (or plain unpack) path. past_seqlens / cos_cache / sin_cache are then
  // never dereferenced and may be null.
  const int hr = h - rotary_offset;
  if (hr < 0 || hr >= rotary_embedding_dim) {
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
    cache_idx = (hr / 2) % half_rotary_embedding_dim;
    sign = (hr % 2 == 0) ? -1 : 1;
    j = (hr % 2 == 0) ? hr + 1 : hr - 1;  // i - sign
  } else {
    cache_idx = hr % half_rotary_embedding_dim;
    sign = (hr < half_rotary_embedding_dim) ? -1 : 1;
    j = (hr + half_rotary_embedding_dim) % rotary_embedding_dim;
  }
  output_data[h] = head_values[h] * cos_data[cache_idx] + sign * head_values[rotary_offset + j] * sin_data[cache_idx];
}

// Launches the fused QK-Norm / rotary prologue. Pass norm_weight == nullptr to disable QK-Norm and
// rotary_embedding_dim == 0 to disable rotary; the caller is responsible for not invoking this when
// both are disabled and the input is not packed.
template <typename T>
Status LaunchQkNormRotaryKernel(cudaStream_t stream, T* output, const T* input, const int32_t* past_seqlens,
                                const int32_t* cumulative_seqlens_q, const T* cos_cache, const T* sin_cache,
                                const T* norm_weight, const float epsilon, const int batch_size,
                                const int token_count, const int num_heads, const int head_size,
                                const int rotary_embedding_dim, const int rotary_offset, const bool interleaved,
                                const int in_seq_stride, const int max_threads_per_block) {
  if (batch_size == 0 || token_count == 0 || num_heads == 0) {
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
  const dim3 grid(token_count, num_heads);
  const dim3 block(tpb);
  QkNormRotaryTNH<<<grid, block, shared_bytes, stream>>>(
      output, input, cos_cache, sin_cache, past_seqlens, cumulative_seqlens_q, norm_weight, epsilon, batch_size,
      head_size, rotary_embedding_dim, rotary_offset, interleaved, in_strides, out_strides);
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
    const int64_t dst_id = static_cast<int64_t>(slot) * kv_hidden_size + hidden_offset;
    // hidden_offset is (kv_head * head_size + channel), which is exactly the PER_CHANNEL scale
    // index. For an unquantized cache the scale pointers are null and this compiles to a copy.
    key_cache[dst_id] =
        QuantizeToCache<T, TCACHE>(key[key_id], GetCacheScale(k_scale, hidden_offset, k_per_channel));
    // In LATENT (MLA) mode there is no separate value tensor or value cache: V is the leading
    // v_head_size channels of the latent row that was just written above.
    if (value_cache != nullptr) {
      const int64_t value_id = static_cast<int64_t>(token_id) * value_stride + hidden_offset;
      value_cache[dst_id] =
          QuantizeToCache<T, TCACHE>(value[value_id], GetCacheScale(v_scale, hidden_offset, v_per_channel));
    }
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

////////// Paged decode attention (flash-decoding style, reads the paged cache in place)
//
// Selected when the static shapes say the step is decode-shaped (token_count == batch_size). Unlike
// the gather-based path it never materializes a dense FP16 copy of the live context: K and V are
// read straight out of their pages and dequantized in registers, so a decode step touches the KV
// cache exactly once at its stored precision.
//
// The shape test is a heuristic, not a proof: token_count == batch_size does not guarantee that
// every sequence contributes exactly one query token (one may contribute two while another
// contributes none). The kernel is therefore indexed by *global query token* and derives both the
// owning sequence and the token's position inside it from cumulative_seqlens_q on device, so it
// stays correct for arbitrary ragged input -- including full prefill. Correctness never depends on
// the host heuristic being right, only speed. See
// docs/contrib_ops/cuda/paged_attention.md section 4.7.
//
// Both scales are folded instead of applied per element, which is exact and free:
//   * K: q'_c = q_c * k_scale_c, so dot(q', k_raw) == dot(q, dequant(k)). PER_TENSOR degenerates to
//     a uniform pre-scale of Q, which is why no separate code path is needed.
//   * V: out_c = (sum_t p_t * v_raw[t][c]) * v_scale_c -- the scale does not depend on t, so it
//     factors out of the accumulation entirely and is applied once in the reduce kernel. It also
//     never enters the softmax denominator.
//
// The KV range of a sequence is split across `num_splits` CTAs; each emits a partial
// (max, denominator, unnormalized accumulator) triple that PagedDecodeReduce combines.

constexpr int kPagedDecodeThreads = 128;
constexpr int kPagedDecodeTile = 128;  // KV tokens scored per iteration
constexpr int kPagedDecodeMaxSplits = 32;

// Cache element -> float. Identical to DequantizeFromCache with a scale of 1, kept separate because
// the decode kernel folds the scales into Q and into the output instead of applying them per read.
template <typename TCACHE>
__device__ __forceinline__ float CacheToFloat(const TCACHE value) {
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (std::is_same<TCACHE, Float8E4M3FN>::value) {
    return value.ToFloat();
  } else  // NOLINT(readability/braces)
#endif
  {
    return static_cast<float>(value);
  }
}

// Number of channel groups the PV accumulation is split into. When head_size < blockDim, several
// groups of `head_size` threads each walk a disjoint subset of the tile's tokens and their partial
// accumulators are summed at the end; this keeps every thread busy and keeps the V reads
// warp-contiguous in the channel dimension.
__host__ __device__ __forceinline__ int PagedDecodeChannelGroups(const int head_size) {
  return head_size >= kPagedDecodeThreads ? 1 : (kPagedDecodeThreads / head_size);
}

template <typename T, typename TCACHE>
__global__ void PagedDecodeSplitKV(const T* __restrict__ query,
                                   const TCACHE* __restrict__ key_cache,
                                   const TCACHE* __restrict__ value_cache,
                                   const float* __restrict__ k_scale,
                                   const int* __restrict__ cumulative_seqlens_q,
                                   const int* __restrict__ cumulative_seqlens_kv,
                                   const int* __restrict__ block_table,
                                   float* __restrict__ partial_out,
                                   float* __restrict__ partial_max,
                                   float* __restrict__ partial_sum,
                                   const int batch_size,
                                   const int num_heads,
                                   const int kv_num_heads,
                                   const int head_size,
                                   const int block_size,
                                   const int max_num_blocks_per_seq,
                                   const int token_count,
                                   const int num_splits,
                                   const float scale,
                                   const float softcap,
                                   const int local_window_size,
                                   const bool k_per_channel) {
  extern __shared__ float paged_decode_smem[];
  const int channel_groups = PagedDecodeChannelGroups(head_size);
  const int acc_elems = channel_groups * head_size;
  // Carve the dynamic block into: Q (head_size), the tile's logits (kPagedDecodeTile), the output
  // accumulator (acc_elems), a block-reduction scratchpad (kPagedDecodeThreads) and the tile's
  // resolved page ids (kPagedDecodeTile ints).
  float* q_sh = paged_decode_smem;
  float* logits_sh = q_sh + head_size;
  float* acc_sh = logits_sh + kPagedDecodeTile;
  float* red_sh = acc_sh + acc_elems;
  int* block_id_sh = reinterpret_cast<int*>(red_sh + kPagedDecodeThreads);

  const int head_id = blockIdx.x;
  const int token_id = blockIdx.y;
  const int split_id = blockIdx.z;
  const int tid = threadIdx.x;

  // One CTA owns one (query token, query head) pair. cumulative_seqlens_q is non-decreasing, so
  // this binary search returns the first sequence whose exclusive end exceeds token_id -- the
  // owning sequence -- and skips sequences that contribute no token at all.
  int left = 0;
  int right = batch_size;
  while (left < right) {
    const int mid = left + (right - left) / 2;
    if (token_id < cumulative_seqlens_q[mid + 1]) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  const int batch_id = left;
  // Defensive: a cumulative_seqlens_q whose total disagrees with token_count would otherwise walk
  // off the end of cumulative_seqlens_kv and block_table. Uniform across the block.
  if (batch_id >= batch_size) {
    return;
  }

  const int64_t partial_head_index =
      (static_cast<int64_t>(split_id) * token_count + token_id) * num_heads + head_id;

  // Causality is resolved per query token instead of assuming one new token per sequence:
  // kv_len - q_len is past_seqlens[batch_id], so the token at offset q_index inside its sequence
  // attends to cached positions [0, past + q_index]. For the decode case (q_len == 1) this reduces
  // to the whole live context, as before.
  const int q_index = token_id - cumulative_seqlens_q[batch_id];
  const int q_len = cumulative_seqlens_q[batch_id + 1] - cumulative_seqlens_q[batch_id];
  const int seq_kv_len = cumulative_seqlens_kv[batch_id + 1] - cumulative_seqlens_kv[batch_id];
  const int kv_len = seq_kv_len - q_len + q_index + 1;

  // Sliding window matches FlashAttention's window_size_left = local_window_size - 1 convention at
  // query position kv_len - 1, i.e. positions in [kv_len - local_window_size, kv_len).
  const int tokens_per_split = (kv_len + num_splits - 1) / num_splits;
  int kv_begin = split_id * tokens_per_split;
  const int kv_end = min(kv_len, kv_begin + tokens_per_split);
  if (local_window_size > 0) {
    kv_begin = max(kv_begin, kv_len - local_window_size);
  }

  if (kv_begin >= kv_end) {
    if (tid == 0) {
      partial_max[partial_head_index] = -FLT_MAX;
      partial_sum[partial_head_index] = 0.0f;
    }
    return;
  }

  const int kv_head_id = head_id / (num_heads / kv_num_heads);
  const int64_t head_offset_in_page = static_cast<int64_t>(kv_head_id) * head_size;
  const int64_t token_stride_in_page = static_cast<int64_t>(kv_num_heads) * head_size;

  const T* q_ptr = query + (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
  for (int c = tid; c < head_size; c += kPagedDecodeThreads) {
    q_sh[c] = static_cast<float>(q_ptr[c]) * GetCacheScale(k_scale, kv_head_id * head_size + c, k_per_channel);
  }
  for (int c = tid; c < acc_elems; c += kPagedDecodeThreads) {
    acc_sh[c] = 0.0f;
  }
  __syncthreads();

  constexpr int kNumWarps = kPagedDecodeThreads / 32;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  // FlashAttention computes softcap as scale_softmax * tanh(qk_raw * softmax_scale / softcap) with
  // scale_softmax == softcap (flash_api.cc), so the effective logit is
  // softcap * tanh(qk * scale / softcap). Match that exactly.
  const float softcap_scale = softcap > 0.0f ? (scale / softcap) : 0.0f;

  float m_state = -FLT_MAX;
  float l_state = 0.0f;

  for (int tile_begin = kv_begin; tile_begin < kv_end; tile_begin += kPagedDecodeTile) {
    const int tile_len = min(kPagedDecodeTile, kv_end - tile_begin);

    // ---- QK: one warp per KV token, 32 lanes cooperating on the head-size dot product ----
    for (int t = warp_id; t < tile_len; t += kNumWarps) {
      const int pos = tile_begin + t;
      const int block_index = pos / block_size;
      const int block_id = block_index < max_num_blocks_per_seq
                               ? block_table[batch_id * max_num_blocks_per_seq + block_index]
                               : -1;
      float dot = 0.0f;
      if (block_id >= 0) {
        const TCACHE* k_ptr = key_cache +
                              (static_cast<int64_t>(block_id) * block_size + (pos % block_size)) * token_stride_in_page +
                              head_offset_in_page;
        for (int c = lane_id; c < head_size; c += 32) {
          dot += q_sh[c] * CacheToFloat<TCACHE>(k_ptr[c]);
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xFFFFFFFFU, dot, offset);
      }
      if (lane_id == 0) {
        block_id_sh[t] = block_id;
        logits_sh[t] = block_id < 0 ? -FLT_MAX
                                    : (softcap > 0.0f ? softcap * tanhf(dot * softcap_scale) : dot * scale);
      }
    }
    __syncthreads();

    // ---- tile max ----
    float local_max = -FLT_MAX;
    for (int t = tid; t < tile_len; t += kPagedDecodeThreads) {
      local_max = fmaxf(local_max, logits_sh[t]);
    }
    red_sh[tid] = local_max;
    __syncthreads();
    for (int stride = kPagedDecodeThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red_sh[tid] = fmaxf(red_sh[tid], red_sh[tid + stride]);
      }
      __syncthreads();
    }
    const float m_tile = red_sh[0];
    __syncthreads();

    // A tile whose pages are all unmapped contributes nothing and would otherwise make the
    // rescale factor exp(-FLT_MAX - -FLT_MAX) == 1 for masked entries.
    if (m_tile == -FLT_MAX) {
      continue;
    }

    const float m_new = fmaxf(m_state, m_tile);
    const float alpha = __expf(m_state - m_new);  // 0 on the first tile (m_state == -FLT_MAX)

    float local_sum = 0.0f;
    for (int t = tid; t < tile_len; t += kPagedDecodeThreads) {
      const float p = __expf(logits_sh[t] - m_new);
      logits_sh[t] = p;
      local_sum += p;
    }
    red_sh[tid] = local_sum;
    __syncthreads();
    for (int stride = kPagedDecodeThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red_sh[tid] += red_sh[tid + stride];
      }
      __syncthreads();
    }
    const float sum_tile = red_sh[0];
    __syncthreads();

    l_state = l_state * alpha + sum_tile;
    m_state = m_new;

    // ---- PV: consecutive threads own consecutive channels so the V reads stay coalesced ----
    if (channel_groups == 1) {
      for (int c = tid; c < head_size; c += kPagedDecodeThreads) {
        float acc = acc_sh[c] * alpha;
        for (int t = 0; t < tile_len; ++t) {
          const int block_id = block_id_sh[t];
          if (block_id < 0) {
            continue;
          }
          const int pos = tile_begin + t;
          const TCACHE* v_ptr = value_cache +
                                (static_cast<int64_t>(block_id) * block_size + (pos % block_size)) * token_stride_in_page +
                                head_offset_in_page;
          acc += logits_sh[t] * CacheToFloat<TCACHE>(v_ptr[c]);
        }
        acc_sh[c] = acc;
      }
    } else if (tid < acc_elems) {
      const int group = tid / head_size;
      const int c = tid - group * head_size;
      float acc = acc_sh[tid] * alpha;
      for (int t = group; t < tile_len; t += channel_groups) {
        const int block_id = block_id_sh[t];
        if (block_id < 0) {
          continue;
        }
        const int pos = tile_begin + t;
        const TCACHE* v_ptr = value_cache +
                              (static_cast<int64_t>(block_id) * block_size + (pos % block_size)) * token_stride_in_page +
                              head_offset_in_page;
        acc += logits_sh[t] * CacheToFloat<TCACHE>(v_ptr[c]);
      }
      acc_sh[tid] = acc;
    }
    __syncthreads();
  }

  const int64_t out_base =
      ((static_cast<int64_t>(split_id) * token_count + token_id) * num_heads + head_id) * head_size;
  for (int c = tid; c < head_size; c += kPagedDecodeThreads) {
    float acc = 0.0f;
    for (int group = 0; group < channel_groups; ++group) {
      acc += acc_sh[group * head_size + c];
    }
    partial_out[out_base + c] = acc;
  }
  if (tid == 0) {
    partial_max[partial_head_index] = m_state;
    partial_sum[partial_head_index] = l_state;
  }
}

// Combine the per-split partials, apply the (folded) V scale and close the softmax. The attention
// sink enters here as one extra logit in the denominator, which is exactly what the FlashAttention
// path's ApplyHeadSink epilogue computes from the log-sum-exp.
template <typename T>
__global__ void PagedDecodeReduce(T* __restrict__ output,
                                  const float* __restrict__ partial_out,
                                  const float* __restrict__ partial_max,
                                  const float* __restrict__ partial_sum,
                                  const float* __restrict__ v_scale,
                                  const T* __restrict__ head_sink,
                                  const int num_heads,
                                  const int kv_num_heads,
                                  const int head_size,
                                  const int token_count,
                                  const int num_splits,
                                  const bool v_per_channel,
                                  const bool use_smooth_softmax) {
  __shared__ float weight_sh[kPagedDecodeMaxSplits];
  __shared__ float max_sh[kPagedDecodeMaxSplits];
  __shared__ float sum_sh[kPagedDecodeMaxSplits];

  const int head_id = blockIdx.x;
  const int token_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t row_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;

  if (tid < num_splits) {
    const int64_t index = (static_cast<int64_t>(tid) * token_count + token_id) * num_heads + head_id;
    max_sh[tid] = partial_max[index];
    sum_sh[tid] = partial_sum[index];
  }
  __syncthreads();

  float m_final = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s) {
    if (sum_sh[s] > 0.0f) {
      m_final = fmaxf(m_final, max_sh[s]);
    }
  }

  if (m_final == -FLT_MAX) {
    for (int c = tid; c < head_size; c += blockDim.x) {
      output[row_base + c] = static_cast<T>(0.0f);
    }
    return;
  }

  float l_final = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float w = sum_sh[s] > 0.0f ? __expf(max_sh[s] - m_final) : 0.0f;
    if (tid == 0) {
      weight_sh[s] = w;
    }
    l_final += sum_sh[s] * w;
  }
  if (use_smooth_softmax) {
    const float sink = head_sink == nullptr ? 0.0f : static_cast<float>(head_sink[head_id]);
    l_final += __expf(sink - m_final);
  }
  __syncthreads();

  const float inv_l = l_final > 0.0f ? (1.0f / l_final) : 0.0f;
  const int kv_head_id = head_id / (num_heads / kv_num_heads);
  for (int c = tid; c < head_size; c += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      if (weight_sh[s] > 0.0f) {
        acc += partial_out[((static_cast<int64_t>(s) * token_count + token_id) * num_heads + head_id) * head_size + c] *
               weight_sh[s];
      }
    }
    output[row_base + c] = static_cast<T>(acc * inv_l *
                                          GetCacheScale(v_scale, kv_head_id * head_size + c, v_per_channel));
  }
}

// Splits are only worth it when there are not enough (query token, head) pairs to fill the device.
// max_kv_len only sizes the launch, so a replay-invariant upper bound is a valid argument: an
// over-estimate costs empty splits that exit after a single device read.
int ComputePagedDecodeSplits(const int token_count, const int num_heads, const int max_kv_len,
                             const int multi_processor_count) {
  const int base_ctas = token_count * num_heads;
  if (base_ctas <= 0 || base_ctas >= 2 * multi_processor_count) {
    return 1;
  }
  const int by_occupancy = (2 * multi_processor_count + base_ctas - 1) / base_ctas;
  const int by_length = (max_kv_len + kPagedDecodeTile - 1) / kPagedDecodeTile;
  return std::max(1, std::min(std::min(by_occupancy, by_length), kPagedDecodeMaxSplits));
}

size_t GetPagedDecodeSharedMemoryBytes(const int head_size) {
  const size_t float_elems = static_cast<size_t>(head_size) + kPagedDecodeTile +
                             static_cast<size_t>(PagedDecodeChannelGroups(head_size)) * head_size +
                             kPagedDecodeThreads;
  return float_elems * sizeof(float) + static_cast<size_t>(kPagedDecodeTile) * sizeof(int);
}

template <typename T, typename TCACHE>
Status LaunchPagedDecodeAttention(const T* query, const TCACHE* key_cache, const TCACHE* value_cache,
                                  const float* k_scale, const float* v_scale,
                                  const bool k_per_channel, const bool v_per_channel,
                                  const int* cumulative_seqlens_q, const int* cumulative_seqlens_kv,
                                  const int* block_table, const T* head_sink, T* output,
                                  float* partial_out, float* partial_max, float* partial_sum,
                                  const int batch_size, const int num_heads, const int kv_num_heads,
                                  const int head_size, const int block_size, const int max_num_blocks_per_seq,
                                  const int token_count, const int num_splits, const float scale,
                                  const float softcap, const int local_window_size,
                                  const bool use_smooth_softmax, cudaStream_t stream) {
  const size_t smem_bytes = GetPagedDecodeSharedMemoryBytes(head_size);
  const dim3 grid(num_heads, token_count, num_splits);
  PagedDecodeSplitKV<T, TCACHE><<<grid, kPagedDecodeThreads, smem_bytes, stream>>>(
      query, key_cache, value_cache, k_scale, cumulative_seqlens_q, cumulative_seqlens_kv, block_table,
      partial_out, partial_max, partial_sum, batch_size, num_heads, kv_num_heads, head_size, block_size,
      max_num_blocks_per_seq, token_count, num_splits, scale, softcap, local_window_size, k_per_channel);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  const dim3 reduce_grid(num_heads, token_count);
  PagedDecodeReduce<T><<<reduce_grid, kPagedDecodeThreads, 0, stream>>>(
      output, partial_out, partial_max, partial_sum, v_scale, head_sink, num_heads, kv_num_heads,
      head_size, token_count, num_splits, v_per_channel, use_smooth_softmax);
  return CUDA_CALL(cudaGetLastError());
}

////////// Unfused paged latent attention (absorbed MLA reference)
//
// Absorbed MLA is MQA with a wide key head and a narrower value head that *aliases* the leading
// v_head_size channels of the same cached row (docs/contrib_ops/cuda/paged_attention.md §12.2):
//
//   key   = [compressed_kv (kv_lora_rank) ; k_pe (qk_rope_head_dim)]   -> head_size    (576 in V3)
//   value = compressed_kv                                             -> v_head_size  (512 in V3)
//
// Neither FlashAttention nor the CUTLASS fMHA wrapper can express that: both cap head_size at 256
// and require v_head_size == head_size. This kernel is the correctness oracle called for in §12.7,
// and it is the only backend for LATENT until a fused MLA kernel lands. It handles arbitrary query
// lengths (prefill, chunked prefill and decode all take the same path), so causality is resolved
// per query token instead of assuming one new token per sequence.
//
// One CTA owns one (query token, query head) pair and streams the whole KV range in tiles, keeping
// an online-softmax (m, l) state and a v_head_size-wide fp32 accumulator in shared memory. The
// quantization scales are folded exactly as in the decode kernel: k_scale into Q at load time, and
// v_scale into the epilogue, so neither ever enters the softmax denominator.

constexpr int kLatentThreads = 256;
constexpr int kLatentTile = 64;  // KV tokens scored per iteration

// Shared memory: Q (head_size) + output accumulator (v_head_size) + tile logits + block reduction
// scratchpad, all fp32, followed by the tile's resolved page ids.
size_t GetPagedLatentSharedMemoryBytes(const int head_size, const int v_head_size) {
  const size_t float_elems = static_cast<size_t>(head_size) + static_cast<size_t>(v_head_size) +
                             kLatentTile + kLatentThreads;
  return float_elems * sizeof(float) + static_cast<size_t>(kLatentTile) * sizeof(int);
}

template <typename T, typename TCACHE>
__global__ void PagedLatentAttentionKernel(const T* __restrict__ query,             // [token, N, head_size]
                                           const TCACHE* __restrict__ key_cache,    // paged, head_size wide
                                           const TCACHE* __restrict__ value_cache,  // == key_cache in LATENT
                                           const float* __restrict__ k_scale,
                                           const float* __restrict__ v_scale,
                                           const int* __restrict__ cumulative_seqlens_q,
                                           const int* __restrict__ past_seqlens,
                                           const int* __restrict__ block_table,
                                           T* __restrict__ output,  // [token, N, v_head_size]
                                           const int batch_size,
                                           const int num_heads,
                                           const int kv_num_heads,
                                           const int head_size,
                                           const int v_head_size,
                                           const int block_size,
                                           const int max_num_blocks_per_seq,
                                           const float scale,
                                           const float softcap,
                                           const int local_window_size,
                                           const bool k_per_channel,
                                           const bool v_per_channel) {
  extern __shared__ float paged_latent_smem[];
  float* q_sh = paged_latent_smem;
  float* acc_sh = q_sh + head_size;
  float* logits_sh = acc_sh + v_head_size;
  float* red_sh = logits_sh + kLatentTile;
  int* block_id_sh = reinterpret_cast<int*>(red_sh + kLatentThreads);

  const int token_id = blockIdx.x;
  const int head_id = blockIdx.y;
  const int tid = threadIdx.x;

  // Locate the sequence this packed token belongs to and its position inside that sequence.
  // cumulative_seqlens_q is a non-decreasing prefix sum, so a binary search is exact for ragged
  // batches, including sequences that contribute zero new tokens.
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
  const int s = token_id - cumulative_seqlens_q[batch_id];

  // Causality: this token's logical position is past_seqlens[b] + s, and it attends every cached
  // position up to and including its own. That is exactly FlashAttention's bottom-right-aligned
  // causal convention for seqlen_k = past + seqlen_q.
  const int kv_end = past_seqlens[batch_id] + s + 1;
  int kv_begin = 0;
  if (local_window_size > 0) {
    // local_window_size counts the current token, matching mha_varlen_fwd's window_size_left = W-1.
    kv_begin = max(0, kv_end - local_window_size);
  }

  const int kv_head_id = head_id / (num_heads / kv_num_heads);
  const int64_t token_stride_in_page = static_cast<int64_t>(kv_num_heads) * head_size;
  const int64_t head_offset_in_page = static_cast<int64_t>(kv_head_id) * head_size;

  // Fold k_scale into Q: dot(q * k_scale, k_raw) == dot(q, dequant(k)), exactly.
  const T* q_ptr = query + (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
  for (int c = tid; c < head_size; c += kLatentThreads) {
    q_sh[c] = static_cast<float>(q_ptr[c]) * GetCacheScale(k_scale, kv_head_id * head_size + c, k_per_channel);
  }
  for (int c = tid; c < v_head_size; c += kLatentThreads) {
    acc_sh[c] = 0.0f;
  }
  __syncthreads();

  constexpr int kNumWarps = kLatentThreads / 32;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  // Match FlashAttention's softcap spelling: softcap * tanh(qk * scale / softcap).
  const float softcap_scale = softcap > 0.0f ? (scale / softcap) : 0.0f;

  float m_state = -FLT_MAX;
  float l_state = 0.0f;

  for (int tile_begin = kv_begin; tile_begin < kv_end; tile_begin += kLatentTile) {
    const int tile_len = min(kLatentTile, kv_end - tile_begin);

    // ---- QK over the full head_size (compressed_kv and k_pe together) ----
    for (int t = warp_id; t < tile_len; t += kNumWarps) {
      const int pos = tile_begin + t;
      const int block_index = pos / block_size;
      const int block_id = block_index < max_num_blocks_per_seq
                               ? block_table[batch_id * max_num_blocks_per_seq + block_index]
                               : -1;
      float dot = 0.0f;
      if (block_id >= 0) {
        const TCACHE* k_ptr = key_cache +
                              (static_cast<int64_t>(block_id) * block_size + (pos % block_size)) * token_stride_in_page +
                              head_offset_in_page;
        for (int c = lane_id; c < head_size; c += 32) {
          dot += q_sh[c] * CacheToFloat<TCACHE>(k_ptr[c]);
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xFFFFFFFFU, dot, offset);
      }
      if (lane_id == 0) {
        block_id_sh[t] = block_id;
        logits_sh[t] = block_id < 0 ? -FLT_MAX
                                    : (softcap > 0.0f ? softcap * tanhf(dot * softcap_scale) : dot * scale);
      }
    }
    __syncthreads();

    // ---- tile max ----
    float local_max = -FLT_MAX;
    for (int t = tid; t < tile_len; t += kLatentThreads) {
      local_max = fmaxf(local_max, logits_sh[t]);
    }
    red_sh[tid] = local_max;
    __syncthreads();
    for (int stride = kLatentThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red_sh[tid] = fmaxf(red_sh[tid], red_sh[tid + stride]);
      }
      __syncthreads();
    }
    const float m_tile = red_sh[0];
    __syncthreads();

    // A tile whose pages are all unmapped contributes nothing; skipping it also avoids the
    // degenerate rescale exp(-FLT_MAX - -FLT_MAX) == 1.
    if (m_tile == -FLT_MAX) {
      continue;
    }

    const float m_new = fmaxf(m_state, m_tile);
    const float alpha = __expf(m_state - m_new);  // 0 on the first contributing tile

    float local_sum = 0.0f;
    for (int t = tid; t < tile_len; t += kLatentThreads) {
      const float p = __expf(logits_sh[t] - m_new);
      logits_sh[t] = p;
      local_sum += p;
    }
    red_sh[tid] = local_sum;
    __syncthreads();
    for (int stride = kLatentThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red_sh[tid] += red_sh[tid + stride];
      }
      __syncthreads();
    }
    const float sum_tile = red_sh[0];
    __syncthreads();

    l_state = l_state * alpha + sum_tile;
    m_state = m_new;

    // ---- PV over the leading v_head_size channels of the same cached rows ----
    for (int c = tid; c < v_head_size; c += kLatentThreads) {
      float acc = acc_sh[c] * alpha;
      for (int t = 0; t < tile_len; ++t) {
        const int block_id = block_id_sh[t];
        if (block_id < 0) {
          continue;
        }
        const int pos = tile_begin + t;
        const TCACHE* v_ptr = value_cache +
                              (static_cast<int64_t>(block_id) * block_size + (pos % block_size)) * token_stride_in_page +
                              head_offset_in_page;
        acc += logits_sh[t] * CacheToFloat<TCACHE>(v_ptr[c]);
      }
      acc_sh[c] = acc;
    }
    __syncthreads();
  }

  const int64_t out_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * v_head_size;
  const float inv_l = (m_state == -FLT_MAX || l_state <= 0.0f) ? 0.0f : (1.0f / l_state);
  for (int c = tid; c < v_head_size; c += kLatentThreads) {
    // V channels are a prefix of the head_size-wide cached row, so a PER_CHANNEL scale is indexed
    // with the head_size stride even though only v_head_size of those channels are read.
    output[out_base + c] = static_cast<T>(acc_sh[c] * inv_l *
                                          GetCacheScale(v_scale, kv_head_id * head_size + c, v_per_channel));
  }
}

template <typename T, typename TCACHE>
Status LaunchPagedLatentAttention(const T* query, const TCACHE* key_cache, const TCACHE* value_cache,
                                  const float* k_scale, const float* v_scale, const bool k_per_channel,
                                  const bool v_per_channel, const int* cumulative_seqlens_q,
                                  const int* past_seqlens, const int* block_table, T* output,
                                  const int batch_size, const int num_heads, const int kv_num_heads,
                                  const int head_size, const int v_head_size, const int block_size,
                                  const int max_num_blocks_per_seq, const int token_count, const float scale,
                                  const float softcap, const int local_window_size, cudaStream_t stream) {
  const size_t smem_bytes = GetPagedLatentSharedMemoryBytes(head_size, v_head_size);
  const dim3 grid(token_count, num_heads);
  PagedLatentAttentionKernel<T, TCACHE><<<grid, kLatentThreads, smem_bytes, stream>>>(
      query, key_cache, value_cache, k_scale, v_scale, cumulative_seqlens_q, past_seqlens, block_table, output,
      batch_size, num_heads, kv_num_heads, head_size, v_head_size, block_size, max_num_blocks_per_seq, scale,
      softcap, local_window_size, k_per_channel, v_per_channel);
  return CUDA_CALL(cudaGetLastError());
}

////////// Launch Kernels

// Prologue shared by every backend: unpack packed QKV, run the fused QK-Norm + rotary kernel when
// requested, and scatter K/V into the paged cache (quantizing on the way in when the cache is
// quantized). None of this depends on which attention backend runs afterwards. On return
// *query_out points at the densified, post-prologue Q.
template <typename T, typename TCACHE>
Status PrepareQueryAndCache(cudaStream_t stream, contrib::PagedAttentionParameters& parameters,
                            PagedAttentionData<T, TCACHE>& data, const int max_threads_per_block,
                            T** query_out) {
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

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

  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);

  if (parameters.do_rotary || parameters.use_qk_norm) {
    // Fused QK-Norm + rotary prologue. Also unpacks Q and K in case of packed_qkv.
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    const int rotary_dim = parameters.do_rotary ? parameters.rotary_dim : 0;
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.q_norm_weight, parameters.qk_norm_epsilon, batch_size, token_count, num_heads, head_size,
        rotary_dim, parameters.rotary_offset, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchQkNormRotaryKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache,
        data.k_norm_weight, parameters.qk_norm_epsilon, batch_size, token_count, kv_num_heads, head_size,
        rotary_dim, parameters.rotary_offset, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
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
  const int key_stride = k_is_packed ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;
  ORT_RETURN_IF_ERROR((LaunchReshapeAndCache<T, TCACHE>(
      key, value, data.key_cache, data.value_cache, data.k_scale, data.v_scale, k_per_channel, v_per_channel,
      const_cast<int*>(data.block_table), past_seqlens, cumulative_seqlens_q, data.slot_mapping, batch_size,
      parameters.max_num_blocks_per_seq, token_count, kv_hidden_size, parameters.block_size,
      parameters.num_blocks, key_stride, value_stride, stream, max_threads_per_block)));

  *query_out = query;
  return Status::OK();
}

// LATENT (absorbed MLA) backend. The latent row is written to the single physical cache by the
// shared prologue (which also applies offset RoPE to the k_pe suffix), then the unfused latent
// kernel reads K and V out of that same cache.
template <typename T, typename TCACHE>
Status LatentAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    float scale) {
  T* query = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQueryAndCache<T, TCACHE>(stream, parameters, data,
                                                       device_prop.maxThreadsPerBlock, &query)));

  // V shares the physical elements of K, so it must be dequantized with the scale those elements
  // were stored with: k_scale, not v_scale (which validation requires to be absent in LATENT).
  ORT_RETURN_IF_ERROR((LaunchPagedLatentAttention<T, TCACHE>(
      query, data.key_cache, /*value_cache*/ data.key_cache, data.k_scale, /*v_scale*/ data.k_scale,
      parameters.k_quant_type == KVQuantizationType::PER_CHANNEL,
      parameters.k_quant_type == KVQuantizationType::PER_CHANNEL,
      data.cumulative_seqlens_q, data.past_seqlens, data.block_table, data.output,
      parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, parameters.head_size,
      parameters.v_head_size, parameters.block_size, parameters.max_num_blocks_per_seq,
      parameters.token_count, scale, parameters.softcap, parameters.local_window_size, stream)));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("latent (MLA) paged attention output", data.output, parameters.token_count, parameters.num_heads,
              parameters.v_head_size);

  return Status::OK();
}

template <typename T, typename TCACHE>
Status PagedDecodeAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    float scale) {
  T* query = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQueryAndCache<T, TCACHE>(stream, parameters, data,
                                                       device_prop.maxThreadsPerBlock, &query)));

  ORT_RETURN_IF_ERROR((LaunchPagedDecodeAttention<T, TCACHE>(
      query, data.key_cache, data.value_cache, data.k_scale, data.v_scale,
      parameters.k_quant_type == KVQuantizationType::PER_CHANNEL,
      parameters.v_quant_type == KVQuantizationType::PER_CHANNEL,
      data.cumulative_seqlens_q, data.cumulative_seqlens_kv, data.block_table, data.head_sink, data.output,
      data.decode_partial_out, data.decode_partial_max, data.decode_partial_sum,
      parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, parameters.head_size,
      parameters.block_size, parameters.max_num_blocks_per_seq, parameters.token_count, data.num_splits,
      scale, parameters.softcap, parameters.local_window_size, parameters.use_smooth_softmax, stream)));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("paged decode attention output", data.output, parameters.token_count, parameters.num_heads,
              parameters.head_size);

  return Status::OK();
}

////////// Paged XQA decode backend
//
// PagedDecodeSplitKV above is a portable scalar kernel: it dequantizes one cache element per
// thread into fp32 and reduces through shared memory, which sustains only a small fraction of
// HBM bandwidth. XQA is the tensor-core decode kernel already used by GroupQueryAttention, and
// TensorRT-LLM's copy of it (contrib_ops/cuda/bert/xqa) supports a paged cache directly. Three
// things have to be reconciled to use it here:
//
//  1. Page size. XQA requires tokensPerPage to divide its CTA tile in the sequence dimension, so
//     the kernels are compiled for kXqaTokensPerPage (128) tokens. PagedAttention's block_size is
//     a graph attribute (256 by default). Because the KV pool is contiguous --
//     [num_blocks, block_size, kv_num_heads, head_size] -- and XQA's PAGED_KV_CACHE_LAYOUT == 1
//     page is exactly [tokens_per_page, kv_num_heads, head_size], block b is bit-for-bit the
//     concatenation of pages [b * pages_per_block, (b + 1) * pages_per_block). ExpandBlockTable-
//     ToPages rewrites blocks larger than 128 tokens accordingly. A 128-token block table is
//     already in XQA page units and passes through without scratch allocation or expansion.
//  2. Per-channel scales. XQA only accepts a scalar dequantization scale per cache. A PER_CHANNEL
//     scale is folded out exactly the same way GroupQueryAttention does it (see the derivation
//     next to LaunchScaleHeadsByChannelScale in group_query_attention_qdq.cuh): k_scale into Q
//     (it multiplies the QK contraction dim) and v_scale into the attention output (it is a free
//     dim of the PV accumulation, so it never touches the softmax denominator).
//  3. Attention sinks. XQA consumes them as fp32, laid out [kv_head][group] -- which is ORT's
//     [num_heads] order -- so only a dtype conversion is needed.

// block_table [batch, max_num_blocks_per_seq] -> page_table [batch, max_num_blocks_per_seq *
// pages_per_block]. An unmapped block (-1) expands to unmapped pages.
__global__ void ExpandBlockTableToPages(const int* __restrict__ block_table,
                                        int* __restrict__ page_table,
                                        const int max_num_blocks_per_seq,
                                        const int pages_per_block,
                                        const int total_pages) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_pages) {
    return;
  }
  const int pages_per_seq = max_num_blocks_per_seq * pages_per_block;
  const int seq = i / pages_per_seq;
  const int page_in_seq = i - seq * pages_per_seq;
  const int block_id = block_table[seq * max_num_blocks_per_seq + page_in_seq / pages_per_block];
  page_table[i] = block_id < 0 ? -1 : block_id * pages_per_block + (page_in_seq % pages_per_block);
}

// Multiply every head vector by a PER_CHANNEL scale indexed [kv_head, channel]. Used to fold
// k_scale into Q before XQA and v_scale into XQA's output afterwards. dst may alias src (the
// output scaling is done in place), so neither pointer is marked __restrict__.
template <typename T>
__global__ void PagedFoldChannelScaleKernel(T* dst,
                                            const T* src,
                                            const float* __restrict__ channel_scale,
                                            const int num_heads, const int head_size,
                                            const int group_size, const int64_t total_elements) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_elements) {
    return;
  }
  const int h = static_cast<int>(i / head_size) % num_heads;
  const int c = static_cast<int>(i % head_size);
  dst[i] = static_cast<T>(static_cast<float>(src[i]) * channel_scale[(h / group_size) * head_size + c]);
}

template <typename T>
__global__ void PagedConvertHeadSinkToFloatKernel(float* __restrict__ dst, const T* __restrict__ src,
                                                  const int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <typename T, typename TCACHE>
Status PagedXqaDecodeAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  T* query = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQueryAndCache<T, TCACHE>(stream, parameters, data, max_threads_per_block, &query)));

  const int pages_per_block = parameters.block_size / onnxruntime::contrib::cuda::kXqaTokensPerPage;
  const int max_pages_per_seq = parameters.max_num_blocks_per_seq * pages_per_block;
  const int* page_table = data.block_table;
  if (pages_per_block > 1) {
    ORT_RETURN_IF_NOT(data.xqa_page_table_scratch, "XQA page-table scratch was not allocated.");
    const int total_pages = batch_size * max_pages_per_seq;
    const int blocks = (total_pages + max_threads_per_block - 1) / max_threads_per_block;
    ExpandBlockTableToPages<<<blocks, max_threads_per_block, 0, stream>>>(
        data.block_table, data.xqa_page_table_scratch,
        parameters.max_num_blocks_per_seq, pages_per_block, total_pages);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    page_table = data.xqa_page_table_scratch;
  }

  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;
  const int64_t q_elements = static_cast<int64_t>(batch_size) * num_heads * head_size;

  if (k_per_channel) {
    // Q may point straight at the (const) graph input when there is no packed-QKV / rotary
    // prologue, so the scaled copy always goes to a dedicated scratch buffer.
    const int blocks = static_cast<int>((q_elements + max_threads_per_block - 1) / max_threads_per_block);
    PagedFoldChannelScaleKernel<T><<<blocks, max_threads_per_block, 0, stream>>>(
        data.xqa_query, query, data.k_scale, num_heads, head_size, num_heads / kv_num_heads, q_elements);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    query = data.xqa_query;
  }

  const float* attention_sinks = nullptr;
  if (parameters.use_smooth_softmax && data.head_sink != nullptr) {
    const int blocks = (num_heads + max_threads_per_block - 1) / max_threads_per_block;
    PagedConvertHeadSinkToFloatKernel<T><<<blocks, max_threads_per_block, 0, stream>>>(
        data.xqa_head_sink, data.head_sink, num_heads);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    attention_sinks = data.xqa_head_sink;
  }

  constexpr bool kIsFp8Cache =
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
      std::is_same<TCACHE, Float8E4M3FN>::value;
#else
      false;
#endif
  constexpr bool kIsInt8Cache = std::is_same<TCACHE, int8_t>::value;
  const XqaQuantType kv_quant_type =
      kIsFp8Cache ? XqaQuantType::kFp8 : (kIsInt8Cache ? XqaQuantType::kInt8 : XqaQuantType::kNone);
  ORT_RETURN_IF_ERROR(LaunchXQAPagedKernel(
      device_prop, stream,
      reinterpret_cast<const void*>(query),
      reinterpret_cast<const void*>(data.key_cache),
      reinterpret_cast<const void*>(data.value_cache),
      reinterpret_cast<void*>(data.output),
      page_table,
      batch_size, num_heads, kv_num_heads, head_size, max_pages_per_seq,
      scale, parameters.local_window_size, data.past_seqlens, attention_sinks,
      // A PER_CHANNEL scale has already been folded into Q / will be applied to the output, so the
      // kernel must use a scale of 1 (which it does when the pointer is null).
      k_per_channel ? nullptr : data.k_scale,
      v_per_channel ? nullptr : data.v_scale,
      kv_quant_type, std::is_same<T, BFloat16>::value,
      data.xqa_workspace, data.xqa_workspace_size));

  if (v_per_channel) {
    const int blocks = static_cast<int>((q_elements + max_threads_per_block - 1) / max_threads_per_block);
    PagedFoldChannelScaleKernel<T><<<blocks, max_threads_per_block, 0, stream>>>(
        data.output, data.output, data.v_scale, num_heads, head_size, num_heads / kv_num_heads, q_elements);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("paged xqa decode attention output", data.output, parameters.token_count, parameters.num_heads,
              parameters.head_size);

  return Status::OK();
}

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
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const float softcap = parameters.softcap;
  bool is_bf16 = std::is_same<T, BFloat16>::value;
  const int local_window_size = parameters.local_window_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int block_size = parameters.block_size;
  // Upper bound on the number of query tokens any one sequence contributes, from paged_attention.cc.
  // mha_varlen_fwd only uses it as `params.seqlen_q` to size the grid in the query dimension; every
  // actual per-sequence length is re-read from cu_seqlens inside the kernel, and an m-block past a
  // sequence's real length exits immediately. An over-estimate therefore costs empty blocks, not
  // correctness (docs/contrib_ops/cuda/paged_attention.md section 4.7).
  const int max_query_len = data.max_query_len;

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across the FA and MEA dispatch paths.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;
  int* block_table = const_cast<int*>(data.block_table);

  T* query = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQueryAndCache<T, TCACHE>(stream, parameters, data, max_threads_per_block, &query)));
  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;

  // Launch kernel
  void* q = reinterpret_cast<void*>(query);
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);

  if constexpr (IsQuantizedCache<TCACHE>::value) {
    // FlashAttention cannot read a quantized page, so dequantize the live context into a dense
    // packed-varlen [total_kv_tokens, kv_num_heads, head_size] buffer (no GQA expansion — Flash
    // does the grouping itself) and use the non-paged varlen entry point.
    ORT_RETURN_IF_ERROR((LaunchGatherAndExpandPagedKVCache<T, TCACHE>(
        data.key_cache, data.value_cache, data.gathered_key, data.gathered_value,
        data.k_scale, data.v_scale, k_per_channel, v_per_channel,
        block_table, cumulative_seqlens_kv, batch_size, /*num_heads*/ kv_num_heads, kv_num_heads,
        head_size, block_size, max_num_blocks_per_seq, data.total_kv_tokens, stream, max_threads_per_block)));

    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
        device_prop, stream, q, reinterpret_cast<void*>(data.gathered_key),
        reinterpret_cast<void*>(data.gathered_value), output, cumulative_seqlens_q, cumulative_seqlens_kv,
        /*seqused_k*/ nullptr, /*block_table*/ nullptr, softmax_lse, batch_size, num_heads, kv_num_heads, head_size,
        max_query_len, data.max_kv_len, token_count, scale, softcap, parameters.is_causal, is_bf16,
        local_window_size - 1, /*max_num_blocks_per_seq*/ 0, /*page_block_size*/ 1,
        data.flash_num_splits, data.flash_softmax_lse_accum, data.flash_out_accum));
  } else {
    void* key_cache = reinterpret_cast<void*>(data.key_cache);
    void* value_cache = reinterpret_cast<void*>(data.value_cache);
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
        device_prop, stream, q, key_cache, value_cache, output, cumulative_seqlens_q, cumulative_seqlens_kv,
        /*seqused_k*/ nullptr, block_table, softmax_lse, batch_size, num_heads, kv_num_heads, head_size,
        max_query_len, data.max_kv_len, token_count, scale, softcap, parameters.is_causal, is_bf16,
        local_window_size - 1,
        max_num_blocks_per_seq, block_size,
        data.flash_num_splits, data.flash_softmax_lse_accum, data.flash_out_accum));
  }

  if (parameters.use_smooth_softmax) {
    // Sink-bearing steps remain unsplit until the split-combine LSE layout is qualified.
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
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const int block_size = parameters.block_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int local_window_size = parameters.local_window_size;
  // Upper bounds from paged_attention.cc, not exact values. total_kv_tokens only sizes the gather
  // (the gather kernel skips indices past the real end of the packed layout) and max_query_len only
  // sizes MEA's `grid_x = ceil_div(sequence_length, kQueriesPerBlock)`; in varlen mode the CUTLASS
  // kernel re-reads num_queries / num_keys from seqstart_q / seqstart_k on device, so a block past
  // a sequence's real length returns without doing any work.
  const int total_kv_tokens = data.total_kv_tokens;
  const int max_query_len = data.max_query_len;

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across FA and MEA dispatch paths.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;
  int* block_table = const_cast<int*>(data.block_table);

  T* query = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQueryAndCache<T, TCACHE>(stream, parameters, data, max_threads_per_block, &query)));
  const bool k_per_channel = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool v_per_channel = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;

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

  // LATENT (MLA) has its own backend: no other kernel can serve v_head_size != head_size over a
  // single aliased cache. Validation guarantees an explicit scale here, so the default above is
  // never the one used.
  if (parameters.is_latent_kv) {
    return LatentAttention(device_prop, stream, parameters, data, scale);
  }

  if (data.use_xqa_decode) {
    return PagedXqaDecodeAttention(device_prop, stream, parameters, data, scale);
  }

  if (data.use_paged_decode) {
    return PagedDecodeAttention(device_prop, stream, parameters, data, scale);
  }

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
