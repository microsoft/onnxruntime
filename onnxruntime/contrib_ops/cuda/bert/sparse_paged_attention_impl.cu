// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/sparse_paged_attention_impl.h"

#include <cfloat>
#include <type_traits>

#include "contrib_ops/cuda/bert/paged_attention_impl.h"

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

template <typename T, typename TCACHE>
__global__ void SparsePagedAttentionKernel(
    const T* query, const TCACHE* key_cache, const TCACHE* value_cache,
    const float* k_scale, const float* v_scale, const int* cumulative_seqlens_q,
    const int* past_seqlens, const int* block_table, const int* selected_indices,
    const int* selected_counts, const T* auxiliary_key, const T* auxiliary_value,
    const int* auxiliary_lengths, const T* head_sink, T* output, int batch_size,
    int token_count, int num_heads, int kv_num_heads, int head_size, int block_size,
    int num_blocks, int max_num_blocks_per_seq, int max_selected_entries,
    int auxiliary_capacity, int auxiliary_num_heads, float scale, float softcap,
    int local_window_size, bool is_causal, bool local_plus_selected,
    bool selected_from_auxiliary, bool auxiliary_kv_shared, bool k_per_channel,
    bool v_per_channel) {
  const int token_id = blockIdx.x;
  const int head_id = blockIdx.y;
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

  extern __shared__ float shared[];
  float* accumulator = shared;
  float* reduction = shared + head_size;
  for (int c = tid; c < head_size; c += blockDim.x) {
    accumulator[c] = 0.0f;
  }

  float running_max = head_sink == nullptr ? -FLT_MAX : SparseActivationValue(head_sink[head_id]);
  float running_sum = head_sink == nullptr ? 0.0f : 1.0f;
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

  for (int candidate = 0; candidate < candidate_count; ++candidate) {
    const bool is_local = candidate < local_count;
    const int selected_slot = candidate - local_count;
    int logical_position = is_local ? local_begin + candidate
                                    : selected_indices[token_id * max_selected_entries + selected_slot];
    if (logical_position < 0) {
      continue;
    }

    const bool use_auxiliary = !is_local && selected_from_auxiliary;
    // local_plus_selected denotes a set union. Avoid weighting a selected main-cache
    // position twice when the external indexer also returns an entry in the local window.
    if (!is_local && !use_auxiliary && local_plus_selected &&
        logical_position >= local_begin && logical_position <= local_end) {
      continue;
    }
    const TCACHE* main_key = nullptr;
    const TCACHE* main_value = nullptr;
    const T* aux_key = nullptr;
    const T* aux_value = nullptr;
    int cache_scale_base = kv_head_id * head_size;

    if (use_auxiliary) {
      if (auxiliary_lengths == nullptr || logical_position >= auxiliary_lengths[batch_id] ||
          logical_position >= auxiliary_capacity) {
        continue;
      }
      const int aux_head = auxiliary_num_heads == 1 ? 0 : kv_head_id;
      const int64_t aux_base =
          ((static_cast<int64_t>(batch_id) * auxiliary_capacity + logical_position) *
               auxiliary_num_heads +
           aux_head) *
          head_size;
      aux_key = auxiliary_key + aux_base;
      aux_value = auxiliary_kv_shared ? aux_key : auxiliary_value + aux_base;
    } else {
      if (logical_position >= main_length || (is_causal && logical_position > query_position)) {
        continue;
      }
      const int logical_block = logical_position / block_size;
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int physical_block =
          block_table[batch_id * max_num_blocks_per_seq + logical_block];
      if (physical_block < 0 || physical_block >= num_blocks) {
        continue;
      }
      const int physical_slot = physical_block * block_size + logical_position % block_size;
      const int64_t cache_base =
          (static_cast<int64_t>(physical_slot) * kv_num_heads + kv_head_id) * head_size;
      main_key = key_cache + cache_base;
      main_value = value_cache + cache_base;
    }

    float dot = 0.0f;
    for (int c = tid; c < head_size; c += blockDim.x) {
      const float key_value =
          use_auxiliary ? SparseActivationValue(aux_key[c])
                        : SparseCacheValue(main_key[c], k_scale, cache_scale_base + c, k_per_channel);
      dot += SparseActivationValue(query[query_base + c]) * key_value;
    }
    reduction[tid] = dot;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (tid < offset) {
        reduction[tid] += reduction[tid + offset];
      }
      __syncthreads();
    }

    float logit = reduction[0] * scale;
    if (softcap > 0.0f) {
      logit = tanhf(logit / softcap) * softcap;
    }
    const float new_max = max(running_max, logit);
    const float old_weight = running_max == -FLT_MAX ? 0.0f : expf(running_max - new_max);
    const float new_weight = expf(logit - new_max);
    for (int c = tid; c < head_size; c += blockDim.x) {
      const float value =
          use_auxiliary ? SparseActivationValue(aux_value[c])
                        : SparseCacheValue(main_value[c], v_scale, cache_scale_base + c, v_per_channel);
      accumulator[c] = accumulator[c] * old_weight + value * new_weight;
    }
    running_sum = running_sum * old_weight + new_weight;
    running_max = new_max;
    __syncthreads();
  }

  const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
  for (int c = tid; c < head_size; c += blockDim.x) {
    output[query_base + c] = static_cast<T>(accumulator[c] * inverse_sum);
  }
}

template <typename T, typename TCACHE>
Status SparseQkvToContext(
    const cudaDeviceProp& device_prop, Stream* stream,
    contrib::PagedAttentionParameters& parameters, PagedAttentionData<T, TCACHE>& data,
    const int* selected_indices, const int* selected_counts, int max_selected_entries,
    const T* auxiliary_key, const T* auxiliary_value, const int* auxiliary_lengths,
    int auxiliary_capacity, int auxiliary_num_heads, SparseAttentionMode attention_mode,
    SelectedKvSource selected_kv_source, bool auxiliary_kv_shared) {
  T* prepared_query = nullptr;
  ORT_RETURN_IF_ERROR(PreparePagedAttentionQueryAndCache<T, TCACHE>(
      device_prop, stream, parameters, data, &prepared_query));

  int threads = 1;
  while (threads < parameters.head_size &&
         threads * 2 <= device_prop.maxThreadsPerBlock) {
    threads *= 2;
  }
  const size_t shared_memory_bytes =
      static_cast<size_t>(parameters.head_size + threads) * sizeof(float);
  if (shared_memory_bytes > static_cast<size_t>(device_prop.sharedMemPerBlock)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "SparsePagedAttention requires ", shared_memory_bytes,
                           " bytes of shared memory, but the device provides ",
                           device_prop.sharedMemPerBlock, " bytes.");
  }

  const dim3 grid(parameters.token_count, parameters.num_heads);
  const float attention_scale =
      parameters.scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(parameters.head_size))
                               : parameters.scale;
  SparsePagedAttentionKernel<T, TCACHE><<<grid, threads, shared_memory_bytes,
                                          static_cast<cudaStream_t>(stream->GetHandle())>>>(
      prepared_query, data.key_cache, data.value_cache, data.k_scale, data.v_scale,
      data.cumulative_seqlens_q, data.past_seqlens, data.block_table, selected_indices,
      selected_counts, auxiliary_key, auxiliary_value, auxiliary_lengths, data.head_sink,
      data.output, parameters.batch_size, parameters.token_count, parameters.num_heads,
      parameters.kv_num_heads, parameters.head_size, parameters.block_size,
      parameters.num_blocks, parameters.max_num_blocks_per_seq, max_selected_entries,
      auxiliary_capacity, auxiliary_num_heads, attention_scale, parameters.softcap,
      parameters.local_window_size, parameters.is_causal,
      attention_mode == SparseAttentionMode::kLocalPlusSelected,
      selected_kv_source == SelectedKvSource::kAuxiliary, auxiliary_kv_shared,
      parameters.k_quant_type == KVQuantizationType::PER_CHANNEL,
      parameters.v_quant_type == KVQuantizationType::PER_CHANNEL);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_SPARSE_PAGED_ATTENTION(T, TCACHE)                        \
  template Status SparseQkvToContext<T, TCACHE>(                             \
      const cudaDeviceProp&, Stream*, contrib::PagedAttentionParameters&,    \
      PagedAttentionData<T, TCACHE>&, const int*, const int*, int, const T*, \
      const T*, const int*, int, int, SparseAttentionMode, SelectedKvSource, \
      bool);

INSTANTIATE_SPARSE_PAGED_ATTENTION(half, half)
INSTANTIATE_SPARSE_PAGED_ATTENTION(BFloat16, BFloat16)
INSTANTIATE_SPARSE_PAGED_ATTENTION(half, int8_t)
INSTANTIATE_SPARSE_PAGED_ATTENTION(BFloat16, int8_t)

#undef INSTANTIATE_SPARSE_PAGED_ATTENTION

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
