// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/sparse_paged_attention_impl.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>

#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/gated_delta_net_mma.cuh"
#include "core/framework/stream_handles.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename TCACHE>
__device__ __forceinline__ float SparseCacheValue(TCACHE value, const float* scale,
                                                  int scale_index, bool per_channel) {
  float result = static_cast<float>(value);
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    result *= scale == nullptr ? 1.0f : scale[per_channel ? scale_index : 0];
  }
  return result;
}

template <typename T>
__device__ __forceinline__ float SparseActivationValue(T value) {
  return static_cast<float>(value);
}

constexpr int kSparsePagedAttentionThreads = 128;
constexpr int kSparsePagedAttentionTile = 128;
constexpr int kSparsePagedAttentionMaxSplits = 32;
constexpr int64_t kSparsePagedAttentionDirectCandidate = int64_t{1} << 62;

__device__ __forceinline__ int64_t SparsePagedAttentionResolveMainCandidate(
    int logical_position, int batch_id, int query_position, int main_length, int past_length,
    const int* cumulative_seqlens_q, const int* block_table, const int* slot_mapping,
    int block_size, int num_blocks, int max_num_blocks_per_seq, bool is_causal) {
  if (logical_position < 0 || logical_position >= main_length ||
      (is_causal && logical_position > query_position)) {
    return -1;
  }
  const int logical_block = logical_position / block_size;
  const int physical_block = logical_block < max_num_blocks_per_seq
                                 ? block_table[static_cast<int64_t>(batch_id) * max_num_blocks_per_seq + logical_block]
                                 : -1;
  if (physical_block < 0 || physical_block >= num_blocks) {
    return -1;
  }
  if (logical_position >= past_length) {
    const int current_token = cumulative_seqlens_q[batch_id] + logical_position - past_length;
    const int64_t mapped_slot = static_cast<int64_t>(physical_block) * block_size + logical_position % block_size;
    const int64_t slot = slot_mapping == nullptr ? mapped_slot : slot_mapping[current_token];
    const int64_t cache_capacity = static_cast<int64_t>(num_blocks) * block_size;
    return slot >= 0 && slot < cache_capacity
               ? kSparsePagedAttentionDirectCandidate + current_token
               : (slot < 0 ? mapped_slot : -1);
  }
  return static_cast<int64_t>(physical_block) * block_size + logical_position % block_size;
}

__host__ __device__ __forceinline__ int SparsePagedAttentionChannelGroups(const int head_size) {
  return head_size >= kSparsePagedAttentionThreads ? 1 : (kSparsePagedAttentionThreads / head_size);
}

template <bool IsMax>
__device__ __forceinline__ float SparsePagedAttentionBlockReduce(float value, float* scratch) {
  static_assert(kSparsePagedAttentionThreads == 128);
  scratch[threadIdx.x] = value;
  __syncthreads();
  const int lane = threadIdx.x % 32;
  // Preserve the shared-memory tree's 64, 32, 16, ... reduction order.
  const float first = IsMax ? fmaxf(scratch[lane], scratch[lane + 64])
                           : scratch[lane] + scratch[lane + 64];
  const float second = IsMax ? fmaxf(scratch[lane + 32], scratch[lane + 96])
                            : scratch[lane + 32] + scratch[lane + 96];
  value = IsMax ? fmaxf(first, second) : first + second;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xFFFFFFFFU, value, offset);
    value = IsMax ? fmaxf(value, other) : value + other;
  }
  value = __shfl_sync(0xFFFFFFFFU, value, 0);
  __syncthreads();
  return value;
}

template <typename T, typename TCACHE, int StaticHeadSize = 0>
__global__ void SparsePagedAttentionSplitKernel(
    const T* query, const T* current_key, const T* current_value,
    const TCACHE* key_cache, const TCACHE* value_cache,
    const float* k_scale, const float* v_scale, const int* cumulative_seqlens_q,
    const int* past_seqlens, const int* block_table, const int* slot_mapping, const int* selected_indices,
    const int* selected_counts, const T* auxiliary_key, const T* auxiliary_value,
    const int* auxiliary_lengths, float* partial_out, float* partial_max,
    float* partial_sum, T* output, const T* head_sink, int batch_size,
    int token_count, int num_heads, int kv_num_heads, int runtime_head_size, int block_size,
    int num_blocks, int max_num_blocks_per_seq, int max_selected_entries,
    int auxiliary_capacity, int auxiliary_num_heads, int current_key_stride,
    int current_value_stride, float scale, float softcap,
    int local_window_size, bool is_causal, bool local_plus_selected,
    bool selected_from_auxiliary, bool auxiliary_kv_shared, bool k_per_channel,
    bool v_per_channel, int num_splits) {
  const int64_t token_head_id = blockIdx.x;
  const int head_id = static_cast<int>(token_head_id % num_heads);
  const int token_id = static_cast<int>(token_head_id / num_heads);
  const int split_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int head_size = StaticHeadSize == 0 ? runtime_head_size : StaticHeadSize;

  int batch_id = 0;
  while (batch_id + 1 < batch_size && token_id >= cumulative_seqlens_q[batch_id + 1]) {
    ++batch_id;
  }
  const int query_index = token_id - cumulative_seqlens_q[batch_id];
  const int query_position = past_seqlens[batch_id] + query_index;
  const int query_length = cumulative_seqlens_q[batch_id + 1] - cumulative_seqlens_q[batch_id];
  const int main_length = past_seqlens[batch_id] + query_length;
  const int kv_head_id = head_id / (num_heads / kv_num_heads);
  const int64_t query_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
  const int64_t partial_head_index =
      (static_cast<int64_t>(split_id) * token_count + token_id) * num_heads + head_id;

  extern __shared__ float shared[];
  const int channel_groups = SparsePagedAttentionChannelGroups(head_size);
  const int accumulator_elements = channel_groups * head_size;
  float* query_shared = shared;
  float* logits = query_shared + head_size;
  float* accumulator = logits + kSparsePagedAttentionTile;
  float* reduction = accumulator + accumulator_elements;
  const int candidate_ref_offset =
      (head_size + kSparsePagedAttentionTile + accumulator_elements + kSparsePagedAttentionThreads + 1) & ~1;
  int64_t* candidate_refs = reinterpret_cast<int64_t*>(shared + candidate_ref_offset);
  for (int c = tid; c < head_size; c += kSparsePagedAttentionThreads) {
    query_shared[c] = SparseActivationValue(query[query_base + c]);
  }
  for (int c = tid; c < accumulator_elements; c += kSparsePagedAttentionThreads) {
    accumulator[c] = 0.0f;
  }
  __syncthreads();

  int local_begin = 0;
  int local_end = -1;
  if (local_plus_selected) {
    local_end = is_causal ? query_position : main_length - 1;
    local_begin = local_window_size > 0 ? max(0, query_position - local_window_size + 1) : 0;
  }

  const int selected_count = min(max(selected_counts[token_id], 0), max_selected_entries);
  const int candidate_count = (local_end >= local_begin ? local_end - local_begin + 1 : 0) + selected_count;
  const int local_count = local_end >= local_begin ? local_end - local_begin + 1 : 0;
  const int candidates_per_split = (candidate_count + num_splits - 1) / num_splits;
  const int candidate_begin = split_id * candidates_per_split;
  const int candidate_end = min(candidate_count, candidate_begin + candidates_per_split);

  if (candidate_begin >= candidate_end) {
    if (num_splits == 1) {
      const int64_t output_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
      for (int c = tid; c < head_size; c += kSparsePagedAttentionThreads) {
        output[output_base + c] = static_cast<T>(0.0f);
      }
      return;
    }
    if (tid == 0) {
      partial_max[partial_head_index] = -FLT_MAX;
      partial_sum[partial_head_index] = 0.0f;
    }
    return;
  }

  constexpr int kNumWarps = kSparsePagedAttentionThreads / 32;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  for (int tile_begin = candidate_begin; tile_begin < candidate_end; tile_begin += kSparsePagedAttentionTile) {
    const int tile_length = min(kSparsePagedAttentionTile, candidate_end - tile_begin);

    for (int tile_offset = warp_id; tile_offset < tile_length; tile_offset += kNumWarps) {
      const int candidate = tile_begin + tile_offset;
      const bool is_local = candidate < local_count;
      const int selected_slot = candidate - local_count;
      const int logical_position =
          is_local ? local_begin + candidate
                   : selected_indices[static_cast<int64_t>(token_id) * max_selected_entries + selected_slot];
      const bool use_auxiliary = !is_local && selected_from_auxiliary;
      int64_t candidate_ref = -1;

      if (logical_position >= 0 &&
          !(!is_local && !use_auxiliary && local_plus_selected &&
            logical_position >= local_begin && logical_position <= local_end)) {
        if (use_auxiliary) {
          if (auxiliary_lengths != nullptr && logical_position < auxiliary_lengths[batch_id] &&
              logical_position < auxiliary_capacity) {
            candidate_ref = -static_cast<int64_t>(logical_position) - 2;
          }
        } else if (logical_position < main_length && (!is_causal || logical_position <= query_position)) {
          if (logical_position >= past_seqlens[batch_id]) {
            const int current_token = cumulative_seqlens_q[batch_id] + logical_position - past_seqlens[batch_id];
            const int logical_block = logical_position / block_size;
            const int physical_block = logical_block < max_num_blocks_per_seq
                                           ? block_table[static_cast<int64_t>(batch_id) * max_num_blocks_per_seq +
                                                         logical_block]
                                           : -1;
            const int64_t cache_capacity = static_cast<int64_t>(num_blocks) * block_size;
            const int64_t mapped_slot = physical_block >= 0
                                            ? static_cast<int64_t>(physical_block) * block_size +
                                                  logical_position % block_size
                                            : -1;
            const int64_t slot = slot_mapping == nullptr ? mapped_slot : slot_mapping[current_token];
            if (physical_block >= 0 && physical_block < num_blocks) {
              if (slot >= 0 && static_cast<int64_t>(slot) < cache_capacity) {
                if constexpr (std::is_same<TCACHE, int8_t>::value) {
                  // Read back the quantized row so current and past tokens have identical semantics.
                  candidate_ref = slot;
                } else {
                  candidate_ref = kSparsePagedAttentionDirectCandidate + current_token;
                }
              } else if (slot < 0 && mapped_slot >= 0 && mapped_slot < cache_capacity) {
                // A suppressed write can denote a prefix-cache hit; read its logical cache row.
                candidate_ref = mapped_slot;
              }
            }
          } else {
            const int logical_block = logical_position / block_size;
            if (logical_block < max_num_blocks_per_seq) {
              const int physical_block =
                  block_table[static_cast<int64_t>(batch_id) * max_num_blocks_per_seq + logical_block];
              if (physical_block >= 0 && physical_block < num_blocks) {
                candidate_ref = static_cast<int64_t>(physical_block) * block_size + logical_position % block_size;
              }
            }
          }
        }
      }

      float dot = 0.0f;
      if (candidate_ref != -1) {
        const bool candidate_is_auxiliary = candidate_ref < -1;
        const bool candidate_is_direct = candidate_ref >= kSparsePagedAttentionDirectCandidate;
        const int64_t candidate_location = candidate_is_auxiliary
                                               ? -candidate_ref - 2
                                               : (candidate_is_direct
                                                      ? candidate_ref - kSparsePagedAttentionDirectCandidate
                                                      : candidate_ref);
        if (candidate_is_auxiliary) {
          const int auxiliary_head = auxiliary_num_heads == 1 ? 0 : kv_head_id;
          const int64_t auxiliary_base =
              ((static_cast<int64_t>(batch_id) * auxiliary_capacity + candidate_location) *
                   auxiliary_num_heads +
               auxiliary_head) *
              head_size;
          for (int c = lane_id; c < head_size; c += 32) {
            dot += query_shared[c] * SparseActivationValue(auxiliary_key[auxiliary_base + c]);
          }
        } else if (candidate_is_direct) {
          const int64_t current_base = candidate_location * current_key_stride + kv_head_id * head_size;
          for (int c = lane_id; c < head_size; c += 32) {
            dot += query_shared[c] * SparseActivationValue(current_key[current_base + c]);
          }
        } else {
          const int64_t cache_base =
              (candidate_location * kv_num_heads + kv_head_id) * head_size;
          const int scale_base = kv_head_id * head_size;
          for (int c = lane_id; c < head_size; c += 32) {
            dot += query_shared[c] * SparseCacheValue(key_cache[cache_base + c], k_scale,
                                                      scale_base + c, k_per_channel);
          }
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xFFFFFFFFU, dot, offset);
      }
      if (lane_id == 0) {
        candidate_refs[tile_offset] = candidate_ref;
        const float scaled_dot = dot * scale;
        logits[tile_offset] = candidate_ref == -1
                                  ? -FLT_MAX
                                  : (softcap > 0.0f ? tanhf(scaled_dot / softcap) * softcap : scaled_dot);
      }
    }
    __syncthreads();

    float tile_max = -FLT_MAX;
    for (int index = tid; index < tile_length; index += kSparsePagedAttentionThreads) {
      tile_max = fmaxf(tile_max, logits[index]);
    }
    tile_max = SparsePagedAttentionBlockReduce<true>(tile_max, reduction);
    if (tile_max == -FLT_MAX) {
      continue;
    }

    const float new_max = fmaxf(running_max, tile_max);
    const float old_weight = __expf(running_max - new_max);
    float local_sum = 0.0f;
    for (int index = tid; index < tile_length; index += kSparsePagedAttentionThreads) {
      const float weight = __expf(logits[index] - new_max);
      logits[index] = weight;
      local_sum += weight;
    }
    const float tile_sum = SparsePagedAttentionBlockReduce<false>(local_sum, reduction);
    running_sum = running_sum * old_weight + tile_sum;
    running_max = new_max;
    __syncthreads();

    if (channel_groups == 1) {
      for (int c = tid; c < head_size; c += 2 * kSparsePagedAttentionThreads) {
        const int second_c = c + kSparsePagedAttentionThreads;
        const bool has_second_channel = second_c < head_size;
        float value_sum = accumulator[c] * old_weight;
        float second_value_sum = has_second_channel ? accumulator[second_c] * old_weight : 0.0f;
        for (int index = 0; index < tile_length; ++index) {
          const int64_t candidate_ref = candidate_refs[index];
          if (candidate_ref == -1) {
            continue;
          }
          const bool candidate_is_auxiliary = candidate_ref < -1;
          const bool candidate_is_direct = candidate_ref >= kSparsePagedAttentionDirectCandidate;
          const int64_t candidate_location = candidate_is_auxiliary
                                                 ? -candidate_ref - 2
                                                 : (candidate_is_direct
                                                        ? candidate_ref - kSparsePagedAttentionDirectCandidate
                                                        : candidate_ref);
          if (candidate_is_auxiliary) {
            const int auxiliary_head = auxiliary_num_heads == 1 ? 0 : kv_head_id;
            const int64_t auxiliary_base =
                ((static_cast<int64_t>(batch_id) * auxiliary_capacity + candidate_location) *
                     auxiliary_num_heads +
                 auxiliary_head) *
                head_size;
            const T* auxiliary_row = auxiliary_kv_shared ? auxiliary_key + auxiliary_base
                                                         : auxiliary_value + auxiliary_base;
            value_sum += logits[index] * SparseActivationValue(auxiliary_row[c]);
            if (has_second_channel) {
              second_value_sum += logits[index] * SparseActivationValue(auxiliary_row[second_c]);
            }
          } else if (candidate_is_direct) {
            const int64_t current_base = candidate_location * current_value_stride + kv_head_id * head_size;
            value_sum += logits[index] * SparseActivationValue(current_value[current_base + c]);
            if (has_second_channel) {
              second_value_sum += logits[index] * SparseActivationValue(current_value[current_base + second_c]);
            }
          } else {
            const int64_t cache_base =
                (candidate_location * kv_num_heads + kv_head_id) * head_size;
            value_sum += logits[index] * SparseCacheValue(value_cache[cache_base + c], v_scale,
                                                          kv_head_id * head_size + c, v_per_channel);
            if (has_second_channel) {
              second_value_sum += logits[index] * SparseCacheValue(value_cache[cache_base + second_c], v_scale,
                                                                   kv_head_id * head_size + second_c, v_per_channel);
            }
          }
        }
        accumulator[c] = value_sum;
        if (has_second_channel) {
          accumulator[second_c] = second_value_sum;
        }
      }
    } else if (tid < accumulator_elements) {
      const int group = tid / head_size;
      const int c = tid - group * head_size;
      float value_sum = accumulator[tid] * old_weight;
      for (int index = group; index < tile_length; index += channel_groups) {
        const int64_t candidate_ref = candidate_refs[index];
        if (candidate_ref == -1) {
          continue;
        }
        const bool candidate_is_auxiliary = candidate_ref < -1;
        const bool candidate_is_direct = candidate_ref >= kSparsePagedAttentionDirectCandidate;
        const int64_t candidate_location = candidate_is_auxiliary
                                               ? -candidate_ref - 2
                                               : (candidate_is_direct
                                                      ? candidate_ref - kSparsePagedAttentionDirectCandidate
                                                      : candidate_ref);
        if (candidate_is_auxiliary) {
          const int auxiliary_head = auxiliary_num_heads == 1 ? 0 : kv_head_id;
          const int64_t auxiliary_base =
              ((static_cast<int64_t>(batch_id) * auxiliary_capacity + candidate_location) *
                   auxiliary_num_heads +
               auxiliary_head) *
              head_size;
          const T* auxiliary_row = auxiliary_kv_shared ? auxiliary_key + auxiliary_base
                                                       : auxiliary_value + auxiliary_base;
          value_sum += logits[index] * SparseActivationValue(auxiliary_row[c]);
        } else if (candidate_is_direct) {
          const int64_t current_base = candidate_location * current_value_stride + kv_head_id * head_size;
          value_sum += logits[index] * SparseActivationValue(current_value[current_base + c]);
        } else {
          const int64_t cache_base =
              (candidate_location * kv_num_heads + kv_head_id) * head_size;
          value_sum += logits[index] * SparseCacheValue(value_cache[cache_base + c], v_scale,
                                                        kv_head_id * head_size + c, v_per_channel);
        }
      }
      accumulator[tid] = value_sum;
    }
    __syncthreads();
  }

  const int64_t output_base = num_splits == 1
                                  ? (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size
                                  : partial_head_index * head_size;
  float inverse_sum = 1.0f;
  if (num_splits == 1) {
    const float final_max = head_sink == nullptr ? running_max : fmaxf(running_max, SparseActivationValue(head_sink[head_id]));
    const float final_sum = running_sum * __expf(running_max - final_max) +
                            (head_sink == nullptr ? 0.0f : __expf(SparseActivationValue(head_sink[head_id]) - final_max));
    inverse_sum = final_sum > 0.0f ? __expf(running_max - final_max) / final_sum : 0.0f;
  }
  for (int c = tid; c < head_size; c += kSparsePagedAttentionThreads) {
    float value_sum = 0.0f;
    for (int group = 0; group < channel_groups; ++group) {
      value_sum += accumulator[group * head_size + c];
    }
    if (num_splits == 1) {
      output[output_base + c] = static_cast<T>(value_sum * inverse_sum);
    } else {
      partial_out[output_base + c] = value_sum;
    }
  }
  if (num_splits > 1 && tid == 0) {
    partial_max[partial_head_index] = running_max;
    partial_sum[partial_head_index] = running_sum;
  }
}

constexpr int kSparsePagedAttentionGqaHeads = 4;
constexpr int kSparsePagedAttentionGqaTile = 64;
constexpr int kSparsePagedAttentionGqaHeadSize = 256;

template <typename T, int SharedStride>
__device__ __forceinline__ void SparsePagedAttentionStageKv(
    T* tile, const T* current, const T* cache, const int64_t* candidate_refs,
    int tile_length, int load_rows, int kv_head, int kv_heads, int current_stride) {
  static_assert(sizeof(T) == 2 && SharedStride % 8 == 0);
  for (int i = threadIdx.x; i < load_rows * 32; i += blockDim.x) {
    const int row = i / 32;
    const int c = (i % 32) * 8;
    const int64_t ref = row < tile_length ? candidate_refs[row] : -1;
    const T* source = nullptr;
    if (ref >= kSparsePagedAttentionDirectCandidate) {
      source = current + (ref - kSparsePagedAttentionDirectCandidate) * current_stride +
               kv_head * kSparsePagedAttentionGqaHeadSize + c;
    } else if (ref >= 0) {
      source = cache + (ref * kv_heads + kv_head) * kSparsePagedAttentionGqaHeadSize + c;
    }
    T* destination = tile + row * SharedStride + c;
    if (source == nullptr) {
      *reinterpret_cast<uint4*>(destination) = uint4{};
    } else if (reinterpret_cast<uintptr_t>(source) % sizeof(uint4) == 0) {
      *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        destination[j] = source[j];
      }
    }
  }
}

template <typename T, int HeadsPerBlock = kSparsePagedAttentionGqaHeads, int DotRows = 1,
          bool TensorCoreQk = false, bool TensorCorePv = false>
__global__ void SparsePagedAttentionGqaTiledKernel(
    const T* query, const T* current_key, const T* current_value,
    const T* key_cache, const T* value_cache, const int* cumulative_seqlens_q,
    const int* past_seqlens, const int* block_table, const int* slot_mapping,
    const int* selected_indices, const int* selected_counts, T* output, const T* head_sink,
    float* partial_out, float* partial_max, float* partial_sum,
    int batch_size, int token_count, int num_heads, int kv_num_heads, int block_size,
    int num_blocks, int max_num_blocks_per_seq, int max_selected_entries,
    int current_key_stride, int current_value_stride, float scale, float softcap, bool is_causal, int num_splits) {
  constexpr int head_size = kSparsePagedAttentionGqaHeadSize;
  constexpr int tile_size = kSparsePagedAttentionGqaTile;
  constexpr int channels_per_lane = head_size / 32;
  constexpr int heads_per_warp = TensorCorePv ? 3 : 1;
  constexpr int head_stride = TensorCorePv ? 4 : 1;
  constexpr int shared_head_stride = head_size + (TensorCoreQk ? 8 : 0);
  constexpr int shared_logit_stride = tile_size + (TensorCoreQk ? 4 : 0);
  __shared__ __align__(32) T kv_tile[tile_size * shared_head_stride];
  __shared__ int64_t candidate_refs[tile_size];
  __shared__ __align__(32) float logits[TensorCoreQk ? 16 : HeadsPerBlock][shared_logit_stride];
  __shared__ __align__(32) T query_tile[TensorCoreQk ? 16 * shared_head_stride : 1];
  T* probabilities = query_tile;
  __shared__ float rescale[TensorCorePv ? 16 : 1];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
#if __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  using MmaT = std::conditional_t<std::is_same<T, half>::value, half, __nv_bfloat16>;
#endif
  float pv[TensorCorePv ? 8 : 1][4] = {};
  static_assert(!TensorCorePv || (TensorCoreQk && std::is_same<T, half>::value));
  const int head_id = blockIdx.x * HeadsPerBlock + warp;
  const int token_id = blockIdx.y;
  const int kv_head_id = head_id / (num_heads / kv_num_heads);
  int batch_id = 0;
  while (batch_id + 1 < batch_size && token_id >= cumulative_seqlens_q[batch_id + 1]) {
    ++batch_id;
  }
  const int query_index = token_id - cumulative_seqlens_q[batch_id];
  const int query_length = cumulative_seqlens_q[batch_id + 1] - cumulative_seqlens_q[batch_id];
  const int query_position = past_seqlens[batch_id] + query_index;
  const int main_length = past_seqlens[batch_id] + query_length;
  const int selected_count = min(max(selected_counts[token_id], 0), max_selected_entries);
  const int candidates_per_split = (selected_count + num_splits - 1) / num_splits;
  const int candidate_begin = blockIdx.z * candidates_per_split;
  const int candidate_end = min(selected_count, candidate_begin + candidates_per_split);
  const int64_t partial_head_index =
      (static_cast<int64_t>(blockIdx.z) * token_count + token_id) * num_heads + head_id;
  const int64_t query_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
  float query_values[channels_per_lane];
  float accumulator[channels_per_lane] = {};
#pragma unroll
  for (int i = 0; i < channels_per_lane; ++i) {
    query_values[i] = SparseActivationValue(query[query_base + lane + i * 32]);
  }
  float running_max[heads_per_warp];
  float running_sum[heads_per_warp] = {};
#pragma unroll
  for (int i = 0; i < heads_per_warp; ++i) {
    running_max[i] = -FLT_MAX;
  }

  for (int tile_begin = candidate_begin; tile_begin < candidate_end; tile_begin += tile_size) {
    const int tile_length = min(tile_size, candidate_end - tile_begin);
    if constexpr (TensorCoreQk) {
      static_assert(HeadsPerBlock == 12);
      for (int i = tid; i < 16 * head_size; i += blockDim.x) {
        const int h = i / head_size;
        const int c = i % head_size;
        query_tile[h * shared_head_stride + c] =
            h < HeadsPerBlock
                ? query[(static_cast<int64_t>(token_id) * num_heads + blockIdx.x * HeadsPerBlock + h) * head_size + c]
                : static_cast<T>(0.0f);
      }
    }
    if (tid < tile_length) {
      const int logical_position =
          selected_indices[static_cast<int64_t>(token_id) * max_selected_entries + tile_begin + tid];
      candidate_refs[tid] = SparsePagedAttentionResolveMainCandidate(
          logical_position, batch_id, query_position, main_length, past_seqlens[batch_id],
          cumulative_seqlens_q, block_table, slot_mapping, block_size, num_blocks,
          max_num_blocks_per_seq, is_causal);
    }
    __syncthreads();
    // Each K/V row is fetched once for query heads sharing the same KV head.
    if constexpr (HeadsPerBlock == 12) {
      SparsePagedAttentionStageKv<T, shared_head_stride>(
          kv_tile, current_key, key_cache, candidate_refs, tile_length, TensorCoreQk ? tile_size : tile_length,
          kv_head_id, kv_num_heads, current_key_stride);
    } else {
      for (int i = tid; i < (TensorCoreQk ? tile_size : tile_length) * head_size; i += blockDim.x) {
        const int row = i / head_size;
        const int c = i % head_size;
        const int64_t ref = row < tile_length ? candidate_refs[row] : -1;
        T value = static_cast<T>(0.0f);
        if (ref >= kSparsePagedAttentionDirectCandidate) {
          value = current_key[(ref - kSparsePagedAttentionDirectCandidate) * current_key_stride +
                              kv_head_id * head_size + c];
        } else if (ref >= 0) {
          value = key_cache[(ref * kv_num_heads + kv_head_id) * head_size + c];
        }
        kv_tile[row * shared_head_stride + c] = value;
      }
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 800
    if constexpr (TensorCoreQk) {
      namespace wmma = nvcuda::wmma;
      using MmaT = std::conditional_t<std::is_same<T, half>::value, half, __nv_bfloat16>;
      static_assert(sizeof(MmaT) == sizeof(T));
      if (warp < 4) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dots;
        wmma::fill_fragment(dots, 0.0f);
        for (int k = 0; k < head_size; k += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, MmaT, wmma::row_major> q;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, MmaT, wmma::col_major> key;
          wmma::load_matrix_sync(q, reinterpret_cast<const MmaT*>(query_tile + k), shared_head_stride);
          wmma::load_matrix_sync(key, reinterpret_cast<const MmaT*>(kv_tile + warp * 16 * shared_head_stride + k), shared_head_stride);
          wmma::mma_sync(dots, q, key, dots);
        }
        wmma::store_matrix_sync(&logits[0][warp * 16], dots, shared_logit_stride, wmma::mem_row_major);
      }
      __syncthreads();
      for (int slot = 0; slot < heads_per_warp; ++slot) {
        const int h = warp + slot * head_stride;
        for (int row = lane; row < tile_length; row += 32) {
          const float scaled_dot = logits[h][row] * scale;
          logits[h][row] = candidate_refs[row] == -1
                               ? -FLT_MAX
                               : (softcap > 0.0f ? tanhf(scaled_dot / softcap) * softcap : scaled_dot);
        }
      }
    } else
#endif
    {
      // Interleave independent rows without changing each dot product's reduction order.
      for (int row_begin = 0; row_begin < tile_length; row_begin += DotRows) {
        float dots[DotRows] = {};
#pragma unroll
        for (int i = 0; i < channels_per_lane; ++i) {
#pragma unroll
          for (int r = 0; r < DotRows; ++r) {
            if (row_begin + r < tile_length) {
              dots[r] += query_values[i] * SparseActivationValue(kv_tile[(row_begin + r) * shared_head_stride + lane + i * 32]);
            }
          }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
          for (int r = 0; r < DotRows; ++r) {
            dots[r] += __shfl_xor_sync(0xFFFFFFFFU, dots[r], offset);
          }
        }
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < DotRows; ++r) {
            const int row = row_begin + r;
            if (row < tile_length) {
              const float scaled_dot = dots[r] * scale;
              logits[warp][row] = candidate_refs[row] == -1
                                      ? -FLT_MAX
                                      : (softcap > 0.0f ? tanhf(scaled_dot / softcap) * softcap : scaled_dot);
            }
          }
        }
      }
    }
    __syncwarp();
    float old_weight[heads_per_warp];
    for (int slot = 0; slot < heads_per_warp; ++slot) {
      const int h = warp + slot * head_stride;
      float tile_max = -FLT_MAX;
      for (int row = lane; row < tile_length; row += 32) {
        tile_max = fmaxf(tile_max, logits[h][row]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFU, tile_max, offset));
      }
      const float new_max = fmaxf(running_max[slot], tile_max);
      old_weight[slot] = __expf(running_max[slot] - new_max);
      float tile_sum = 0.0f;
      for (int row = lane; row < tile_length; row += 32) {
        const float weight = candidate_refs[row] == -1 ? 0.0f : __expf(logits[h][row] - new_max);
        logits[h][row] = weight;
        tile_sum += weight;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_sum += __shfl_xor_sync(0xFFFFFFFFU, tile_sum, offset);
      }
      running_sum[slot] = running_sum[slot] * old_weight[slot] + tile_sum;
      running_max[slot] = new_max;
      if constexpr (TensorCorePv) {
        if (lane == 0) {
          rescale[h] = old_weight[slot];
        }
      }
    }
    __syncthreads();
    if constexpr (TensorCorePv) {
      if (tid >= HeadsPerBlock && tid < 16) {
        rescale[tid] = 1.0f;
      }
      for (int i = tid; i < 16 * tile_size; i += blockDim.x) {
        const int h = i / tile_size;
        const int row = i % tile_size;
        probabilities[i] = h < HeadsPerBlock && row < tile_length
                               ? static_cast<T>(logits[h][row])
                               : static_cast<T>(0.0f);
      }
    }
    // Reuse the key tile's storage after all warps have finished their dot products.
    if constexpr (HeadsPerBlock == 12) {
      SparsePagedAttentionStageKv<T, shared_head_stride>(
          kv_tile, current_value, value_cache, candidate_refs, tile_length, TensorCorePv ? tile_size : tile_length,
          kv_head_id, kv_num_heads, current_value_stride);
    } else {
      for (int i = tid; i < (TensorCorePv ? tile_size : tile_length) * head_size; i += blockDim.x) {
        const int row = i / head_size;
        const int c = i % head_size;
        const int64_t ref = row < tile_length ? candidate_refs[row] : -1;
        T value = static_cast<T>(0.0f);
        if (ref >= kSparsePagedAttentionDirectCandidate) {
          value = current_value[(ref - kSparsePagedAttentionDirectCandidate) * current_value_stride +
                                kv_head_id * head_size + c];
        } else if (ref >= 0) {
          value = value_cache[(ref * kv_num_heads + kv_head_id) * head_size + c];
        }
        kv_tile[row * shared_head_stride + c] = value;
      }
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 800
    if constexpr (TensorCorePv) {
      const int h = lane / 4;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        pv[n][0] *= rescale[h];
        pv[n][1] *= rescale[h];
        pv[n][2] *= rescale[h + 8];
        pv[n][3] *= rescale[h + 8];
      }
      for (int k = 0; k < tile_size; k += 16) {
        uint32_t p[4];
        gated_delta_net::LoadFragA<false>(p, probabilities, tile_size, 0, k, lane);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          uint32_t value[2];
          gated_delta_net::LoadFragB<false>(value, kv_tile, shared_head_stride, k, warp * 64 + n * 8, lane);
          gated_delta_net::MmaM16N8K16(pv[n], p, value);
        }
      }
    } else
#endif
    {
#pragma unroll
      for (int i = 0; i < channels_per_lane; ++i) {
        accumulator[i] *= old_weight[0];
      }
      for (int row = 0; row < tile_length; ++row) {
        const float weight = logits[warp][row];
#pragma unroll
        for (int i = 0; i < channels_per_lane; ++i) {
          accumulator[i] += weight * SparseActivationValue(kv_tile[row * shared_head_stride + lane + i * 32]);
        }
      }
    }
    __syncthreads();
  }

  float inverse_sum[heads_per_warp];
  for (int slot = 0; slot < heads_per_warp; ++slot) {
    inverse_sum[slot] = 1.0f;
    const int h = head_id + slot * head_stride;
    if (num_splits == 1) {
      const float final_max = head_sink == nullptr
                                  ? running_max[slot]
                                  : fmaxf(running_max[slot], SparseActivationValue(head_sink[h]));
      const float final_sum = running_sum[slot] * __expf(running_max[slot] - final_max) +
                              (head_sink == nullptr ? 0.0f : __expf(SparseActivationValue(head_sink[h]) - final_max));
      inverse_sum[slot] = final_sum > 0.0f ? __expf(running_max[slot] - final_max) / final_sum : 0.0f;
    }
  }
#if __CUDA_ARCH__ >= 800
  if constexpr (TensorCorePv) {
    if (lane == 0) {
      for (int slot = 0; slot < heads_per_warp; ++slot) {
        rescale[warp + slot * head_stride] = inverse_sum[slot];
      }
    }
    __syncthreads();
    const int h = lane / 4;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int c = warp * 64 + n * 8 + (lane % 4) * 2;
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        const int head = h + row * 8;
        if (head < HeadsPerBlock) {
          const int64_t base = (static_cast<int64_t>(token_id) * num_heads + blockIdx.x * HeadsPerBlock + head) * head_size;
          output[base + c] = static_cast<T>(pv[n][row * 2] * rescale[head]);
          output[base + c + 1] = static_cast<T>(pv[n][row * 2 + 1] * rescale[head]);
        }
      }
    }
  } else
#endif
  {
#pragma unroll
    for (int i = 0; i < channels_per_lane; ++i) {
      if (num_splits == 1) {
        output[query_base + lane + i * 32] = static_cast<T>(accumulator[i] * inverse_sum[0]);
      } else {
        partial_out[partial_head_index * head_size + lane + i * 32] = accumulator[i];
      }
    }
    if (num_splits > 1 && lane == 0) {
      partial_max[partial_head_index] = running_max[0];
      partial_sum[partial_head_index] = running_sum[0];
    }
  }
}

template <typename T>
__global__ void SparsePagedAttentionReduceKernel(
    T* output, const float* partial_out, const float* partial_max,
    const float* partial_sum, const T* head_sink, int token_count,
    int num_heads, int head_size, int num_splits) {
  __shared__ float weights[kSparsePagedAttentionMaxSplits];
  __shared__ float maxima[kSparsePagedAttentionMaxSplits];
  __shared__ float sums[kSparsePagedAttentionMaxSplits];
  const int64_t token_head_id = blockIdx.x;
  const int head_id = static_cast<int>(token_head_id % num_heads);
  const int token_id = static_cast<int>(token_head_id / num_heads);
  const int tid = threadIdx.x;

  if (tid < num_splits) {
    const int64_t index = (static_cast<int64_t>(tid) * token_count + token_id) * num_heads + head_id;
    maxima[tid] = partial_max[index];
    sums[tid] = partial_sum[index];
  }
  __syncthreads();

  float final_max = head_sink == nullptr ? -FLT_MAX : SparseActivationValue(head_sink[head_id]);
  for (int split = 0; split < num_splits; ++split) {
    if (sums[split] > 0.0f) {
      final_max = fmaxf(final_max, maxima[split]);
    }
  }

  float final_sum = head_sink == nullptr ? 0.0f : __expf(SparseActivationValue(head_sink[head_id]) - final_max);
  for (int split = 0; split < num_splits; ++split) {
    const float weight = sums[split] > 0.0f ? __expf(maxima[split] - final_max) : 0.0f;
    if (tid == 0) {
      weights[split] = weight;
    }
    final_sum += sums[split] * weight;
  }
  __syncthreads();

  const int64_t output_base = (static_cast<int64_t>(token_id) * num_heads + head_id) * head_size;
  const float inverse_sum = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
  for (int c = tid; c < head_size; c += kSparsePagedAttentionThreads) {
    float value_sum = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      if (weights[split] > 0.0f) {
        const int64_t partial_base =
            ((static_cast<int64_t>(split) * token_count + token_id) * num_heads + head_id) * head_size;
        value_sum += partial_out[partial_base + c] * weights[split];
      }
    }
    output[output_base + c] = static_cast<T>(value_sum * inverse_sum);
  }
}

int SparsePagedAttentionHeadsPerBlock(
    const contrib::PagedAttentionParameters& parameters,
    SparseAttentionMode attention_mode, SelectedKvSource selected_kv_source,
    bool matching_cache_type, size_t shared_memory_per_block) {
  constexpr size_t tiled_shared_bytes =
      kSparsePagedAttentionGqaTile * kSparsePagedAttentionGqaHeadSize * sizeof(half) +
      kSparsePagedAttentionGqaTile * sizeof(int64_t) +
      kSparsePagedAttentionGqaHeads * kSparsePagedAttentionGqaTile * sizeof(float);
  return matching_cache_type && parameters.head_size == kSparsePagedAttentionGqaHeadSize &&
                 (parameters.num_heads / parameters.kv_num_heads) % kSparsePagedAttentionGqaHeads == 0 &&
                 attention_mode == SparseAttentionMode::kSelectedOnly && selected_kv_source == SelectedKvSource::kMain &&
                 tiled_shared_bytes <= shared_memory_per_block
             ? kSparsePagedAttentionGqaHeads
             : 1;
}

int ComputeSparsePagedAttentionSplits(
    const int token_count, const int num_heads, const int max_candidate_count,
    const int multi_processor_count, const int heads_per_block) {
  const int64_t base_blocks = static_cast<int64_t>(token_count) * (num_heads / heads_per_block);
  if (base_blocks <= 0 || base_blocks >= 2 * multi_processor_count) {
    return 1;
  }
  const int target_waves =
      heads_per_block == kSparsePagedAttentionGqaHeads && token_count <= 8 ? 4 : 2;
  const int by_occupancy =
      static_cast<int>((target_waves * multi_processor_count + base_blocks - 1) / base_blocks);
  const int tile_size = heads_per_block == kSparsePagedAttentionGqaHeads
                            ? kSparsePagedAttentionGqaTile
                            : kSparsePagedAttentionTile;
  const int by_length = (max_candidate_count + tile_size - 1) / tile_size;
  return max(1, min(min(by_occupancy, by_length), kSparsePagedAttentionMaxSplits));
}

size_t GetSparsePagedAttentionSharedMemoryBytes(const int head_size) {
  const size_t float_elements = static_cast<size_t>(head_size) + kSparsePagedAttentionTile +
                                static_cast<size_t>(SparsePagedAttentionChannelGroups(head_size)) * head_size +
                                kSparsePagedAttentionThreads;
  const size_t aligned_float_elements = (float_elements + 1) & ~size_t{1};
  return aligned_float_elements * sizeof(float) + static_cast<size_t>(kSparsePagedAttentionTile) * sizeof(int64_t);
}

template <typename T, typename TCACHE>
Status SparseQkvToContext(
    const cudaDeviceProp& device_prop, Stream* stream,
    contrib::PagedAttentionParameters& parameters, PagedAttentionData<T, TCACHE>& data,
    const int* selected_indices, const int* selected_counts, int max_selected_entries,
    const T* auxiliary_key, const T* auxiliary_value, const int* auxiliary_lengths,
    int auxiliary_capacity, int auxiliary_num_heads, SparseAttentionMode attention_mode,
    SelectedKvSource selected_kv_source, bool auxiliary_kv_shared,
    float* partial_out, float* partial_max, float* partial_sum, int num_splits) {
  const size_t shared_memory_bytes = GetSparsePagedAttentionSharedMemoryBytes(parameters.head_size);
  if (shared_memory_bytes > static_cast<size_t>(device_prop.sharedMemPerBlock)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "SparsePagedAttention requires ", shared_memory_bytes,
                           " bytes of shared memory, but the device provides ",
                           device_prop.sharedMemPerBlock, " bytes.");
  }
  const int64_t token_head_count = static_cast<int64_t>(parameters.token_count) * parameters.num_heads;
  if (token_head_count > device_prop.maxGridSize[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "SparsePagedAttention requires ", token_head_count,
                           " token-head blocks, but the device grid X dimension supports ",
                           device_prop.maxGridSize[0], ".");
  }

  T* prepared_query = nullptr;
  T* prepared_key = nullptr;
  T* prepared_value = nullptr;
  int prepared_key_stride = 0;
  int prepared_value_stride = 0;
  ORT_RETURN_IF_ERROR((PreparePagedAttentionQueryAndCache<T, TCACHE>(
      device_prop, stream, parameters, data, &prepared_query, &prepared_key, &prepared_value,
      &prepared_key_stride, &prepared_value_stride)));

  const dim3 grid(static_cast<unsigned int>(token_head_count), num_splits);
  const float attention_scale =
      parameters.scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(parameters.head_size))
                               : parameters.scale;
  if constexpr (std::is_same<T, TCACHE>::value) {
    if (SparsePagedAttentionHeadsPerBlock(parameters, attention_mode, selected_kv_source, true,
                                          device_prop.sharedMemPerBlock) == kSparsePagedAttentionGqaHeads) {
      constexpr size_t twelve_head_shared_bytes =
          kSparsePagedAttentionGqaTile * kSparsePagedAttentionGqaHeadSize * sizeof(T) +
          kSparsePagedAttentionGqaTile * sizeof(int64_t) +
          12 * kSparsePagedAttentionGqaTile * sizeof(float);
      constexpr size_t six_head_shared_bytes =
          kSparsePagedAttentionGqaTile * kSparsePagedAttentionGqaHeadSize * sizeof(T) +
          kSparsePagedAttentionGqaTile * sizeof(int64_t) +
          6 * kSparsePagedAttentionGqaTile * sizeof(float);
      const bool use_twelve_heads = parameters.token_count > 64 && num_splits == 1 &&
                                    (parameters.num_heads / parameters.kv_num_heads) % 12 == 0 &&
                                    twelve_head_shared_bytes <= device_prop.sharedMemPerBlock;
      const bool use_six_heads = parameters.token_count > 64 && num_splits == 1 &&
                                 (parameters.num_heads / parameters.kv_num_heads) % 6 == 0 &&
                                 six_head_shared_bytes <= device_prop.sharedMemPerBlock;
      const int heads_per_block = use_twelve_heads ? 12 : (use_six_heads ? 6 : kSparsePagedAttentionGqaHeads);
      const dim3 tiled_grid(parameters.num_heads / heads_per_block, parameters.token_count, num_splits);
      static const bool enable_tensor_core_qk =
          ParseEnvironmentVariableWithDefault<bool>("ORT_SPARSE_PREFILL_TENSOR_CORE_QK", false);
      static const bool enable_tensor_core_pv =
          ParseEnvironmentVariableWithDefault<bool>("ORT_SPARSE_PREFILL_TENSOR_CORE_PV", false);
      const auto tensor_core_pv_kernel = [] {
        if constexpr (std::is_same<T, half>::value) {
          return SparsePagedAttentionGqaTiledKernel<T, 12, 4, true, true>;
        } else {
          return SparsePagedAttentionGqaTiledKernel<T, 12, 4, true>;
        }
      }();
      bool use_tensor_core_qk = false;
      bool use_tensor_core_pv = false;
      if (enable_tensor_core_qk && use_twelve_heads && device_prop.major >= 8) {
        cudaFuncAttributes attributes{};
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaFuncGetAttributes(&attributes, SparsePagedAttentionGqaTiledKernel<T, 12, 4, true>)));
        use_tensor_core_qk = attributes.ptxVersion >= 80 &&
                             attributes.sharedSizeBytes <= device_prop.sharedMemPerBlock;
        if (use_tensor_core_qk && enable_tensor_core_pv && std::is_same<T, half>::value) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaFuncGetAttributes(&attributes, tensor_core_pv_kernel)));
          use_tensor_core_pv = attributes.sharedSizeBytes <= device_prop.sharedMemPerBlock;
        }
      }
      const auto tiled_kernel = use_twelve_heads
                                    ? (use_tensor_core_pv   ? tensor_core_pv_kernel
                                       : use_tensor_core_qk ? SparsePagedAttentionGqaTiledKernel<T, 12, 4, true>
                                                            : SparsePagedAttentionGqaTiledKernel<T, 12, 4>)
                                    : (use_six_heads ? SparsePagedAttentionGqaTiledKernel<T, 6>
                                                     : SparsePagedAttentionGqaTiledKernel<T>);
      tiled_kernel<<<tiled_grid, (use_tensor_core_pv ? 4 : heads_per_block) * 32, 0,
                     static_cast<cudaStream_t>(stream->GetHandle())>>>(
          prepared_query, prepared_key, prepared_value, data.key_cache, data.value_cache,
          data.cumulative_seqlens_q, data.past_seqlens, data.block_table, data.slot_mapping,
          selected_indices, selected_counts, data.output, data.head_sink,
          partial_out, partial_max, partial_sum,
          parameters.batch_size, parameters.token_count, parameters.num_heads, parameters.kv_num_heads, parameters.block_size,
          parameters.num_blocks, parameters.max_num_blocks_per_seq, max_selected_entries,
          prepared_key_stride, prepared_value_stride, attention_scale, parameters.softcap, parameters.is_causal, num_splits);
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
      if (num_splits > 1) {
        const dim3 reduce_grid(parameters.num_heads, parameters.token_count);
        SparsePagedAttentionReduceKernel<T><<<reduce_grid, kSparsePagedAttentionThreads, 0,
                                              static_cast<cudaStream_t>(stream->GetHandle())>>>(
            data.output, partial_out, partial_max, partial_sum, data.head_sink,
            parameters.token_count, parameters.num_heads, parameters.head_size, num_splits);
      }
      return CUDA_CALL(cudaGetLastError());
    }
  }
  const auto kernel = parameters.head_size == 256
                          ? SparsePagedAttentionSplitKernel<T, TCACHE, 256>
                          : SparsePagedAttentionSplitKernel<T, TCACHE>;
  kernel<<<grid, kSparsePagedAttentionThreads, shared_memory_bytes,
           static_cast<cudaStream_t>(stream->GetHandle())>>>(
      prepared_query, prepared_key, prepared_value, data.key_cache, data.value_cache, data.k_scale, data.v_scale,
      data.cumulative_seqlens_q, data.past_seqlens, data.block_table, data.slot_mapping, selected_indices,
      selected_counts, auxiliary_key, auxiliary_value, auxiliary_lengths, partial_out,
      partial_max, partial_sum, data.output, data.head_sink, parameters.batch_size, parameters.token_count, parameters.num_heads,
      parameters.kv_num_heads, parameters.head_size, parameters.block_size,
      parameters.num_blocks, parameters.max_num_blocks_per_seq, max_selected_entries,
      auxiliary_capacity, auxiliary_num_heads, prepared_key_stride, prepared_value_stride,
      attention_scale, parameters.softcap,
      parameters.local_window_size, parameters.is_causal,
      attention_mode == SparseAttentionMode::kLocalPlusSelected,
      selected_kv_source == SelectedKvSource::kAuxiliary, auxiliary_kv_shared,
      parameters.k_quant_type == KVQuantizationType::PER_CHANNEL,
      parameters.v_quant_type == KVQuantizationType::PER_CHANNEL, num_splits);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  if (num_splits > 1) {
    const dim3 reduce_grid(static_cast<unsigned int>(token_head_count));
    SparsePagedAttentionReduceKernel<T><<<reduce_grid, kSparsePagedAttentionThreads, 0,
                                          static_cast<cudaStream_t>(stream->GetHandle())>>>(
        data.output, partial_out, partial_max, partial_sum, data.head_sink,
        parameters.token_count, parameters.num_heads, parameters.head_size, num_splits);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_SPARSE_PAGED_ATTENTION(T, TCACHE)                        \
  template Status SparseQkvToContext<T, TCACHE>(                             \
      const cudaDeviceProp&, Stream*, contrib::PagedAttentionParameters&,    \
      PagedAttentionData<T, TCACHE>&, const int*, const int*, int, const T*, \
      const T*, const int*, int, int, SparseAttentionMode, SelectedKvSource, \
      bool, float*, float*, float*, int);

INSTANTIATE_SPARSE_PAGED_ATTENTION(half, half)
INSTANTIATE_SPARSE_PAGED_ATTENTION(BFloat16, BFloat16)
INSTANTIATE_SPARSE_PAGED_ATTENTION(half, int8_t)
INSTANTIATE_SPARSE_PAGED_ATTENTION(BFloat16, int8_t)

#undef INSTANTIATE_SPARSE_PAGED_ATTENTION

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
