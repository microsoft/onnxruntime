// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/sparse_paged_attention_impl.h"

#include <cfloat>
#include <type_traits>

#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "core/framework/stream_handles.h"
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

__host__ __device__ __forceinline__ int SparsePagedAttentionChannelGroups(const int head_size) {
  return head_size >= kSparsePagedAttentionThreads ? 1 : (kSparsePagedAttentionThreads / head_size);
}

template <typename T, typename TCACHE>
__global__ void SparsePagedAttentionSplitKernel(
    const T* query, const T* current_key, const T* current_value,
    const TCACHE* key_cache, const TCACHE* value_cache,
    const float* k_scale, const float* v_scale, const int* cumulative_seqlens_q,
    const int* past_seqlens, const int* block_table, const int* slot_mapping, const int* selected_indices,
    const int* selected_counts, const T* auxiliary_key, const T* auxiliary_value,
    const int* auxiliary_lengths, float* partial_out, float* partial_max,
    float* partial_sum, T* output, const T* head_sink, int batch_size,
    int token_count, int num_heads, int kv_num_heads, int head_size, int block_size,
    int num_blocks, int max_num_blocks_per_seq, int max_selected_entries,
    int auxiliary_capacity, int auxiliary_num_heads, int current_key_stride,
    int current_value_stride, float scale, float softcap,
    int local_window_size, bool is_causal, bool local_plus_selected,
    bool selected_from_auxiliary, bool auxiliary_kv_shared, bool k_per_channel,
    bool v_per_channel, int num_splits) {
  const int head_id = blockIdx.x;
  const int token_id = blockIdx.y;
  const int split_id = blockIdx.z;
  const int tid = threadIdx.x;

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
            const int slot = slot_mapping == nullptr
                                 ? (physical_block >= 0
                                        ? physical_block * block_size + logical_position % block_size
                                        : -1)
                                 : slot_mapping[current_token];
            if (physical_block >= 0 && physical_block < num_blocks &&
                slot >= 0 && slot < num_blocks * block_size) {
              candidate_ref = kSparsePagedAttentionDirectCandidate + current_token;
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
    reduction[tid] = tile_max;
    __syncthreads();
    for (int stride = kSparsePagedAttentionThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    tile_max = reduction[0];
    __syncthreads();
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
    reduction[tid] = local_sum;
    __syncthreads();
    for (int stride = kSparsePagedAttentionThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    running_sum = running_sum * old_weight + reduction[0];
    running_max = new_max;
    __syncthreads();

    if (channel_groups == 1) {
      for (int c = tid; c < head_size; c += kSparsePagedAttentionThreads) {
        float value_sum = accumulator[c] * old_weight;
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
        accumulator[c] = value_sum;
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

template <typename T>
__global__ void SparsePagedAttentionReduceKernel(
    T* output, const float* partial_out, const float* partial_max,
    const float* partial_sum, const T* head_sink, int token_count,
    int num_heads, int head_size, int num_splits) {
  __shared__ float weights[kSparsePagedAttentionMaxSplits];
  __shared__ float maxima[kSparsePagedAttentionMaxSplits];
  __shared__ float sums[kSparsePagedAttentionMaxSplits];
  const int head_id = blockIdx.x;
  const int token_id = blockIdx.y;
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

int ComputeSparsePagedAttentionSplits(
    const int token_count, const int num_heads, const int max_candidate_count,
    const int multi_processor_count) {
  const int64_t base_blocks = static_cast<int64_t>(token_count) * num_heads;
  if (base_blocks <= 0 || base_blocks >= 2 * multi_processor_count) {
    return 1;
  }
  const int by_occupancy = static_cast<int>((2 * multi_processor_count + base_blocks - 1) / base_blocks);
  const int by_length = (max_candidate_count + kSparsePagedAttentionTile - 1) / kSparsePagedAttentionTile;
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

  T* prepared_query = nullptr;
  T* prepared_key = nullptr;
  T* prepared_value = nullptr;
  int prepared_key_stride = 0;
  int prepared_value_stride = 0;
  ORT_RETURN_IF_ERROR((PreparePagedAttentionQueryAndCache<T, TCACHE>(
      device_prop, stream, parameters, data, &prepared_query, &prepared_key, &prepared_value,
      &prepared_key_stride, &prepared_value_stride)));

  const dim3 grid(parameters.num_heads, parameters.token_count, num_splits);
  const float attention_scale =
      parameters.scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(parameters.head_size))
                               : parameters.scale;
  SparsePagedAttentionSplitKernel<T, TCACHE><<<grid, kSparsePagedAttentionThreads, shared_memory_bytes,
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
    const dim3 reduce_grid(parameters.num_heads, parameters.token_count);
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
