// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <limits>
#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kSplitCandidateSize = 256;
constexpr int kTargetSplitBlocks = 1024;
constexpr size_t kMaxFusedSharedBytes = 16 * 1024;
constexpr size_t kMaxValidationSharedBytes = 32 * 1024;
constexpr int kGroupedPrefillCandidatesPerTile = 8;
constexpr int kGroupedPrefillQueryHeadsPerBlock = 6;
constexpr int kGroupedDecodeQueryHeadsPerBlock = 8;
constexpr int kGroupedDecodeCandidatesPerSplit = 64;
constexpr int kTargetGroupedDecodeBlocks = 128;

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

__device__ __forceinline__ uint32_t PackMmaHalf2(half low, half high) {
  return static_cast<uint32_t>(__half_as_ushort(low)) |
         (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

__device__ __forceinline__ void MmaM16N8K16(float (&output)[4],
                                            const uint32_t (&query_fragment)[4],
                                            const uint32_t (&key_fragment)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
      : "r"(query_fragment[0]), "r"(query_fragment[1]),
        "r"(query_fragment[2]), "r"(query_fragment[3]),
        "r"(key_fragment[0]), "r"(key_fragment[1]));
#else
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(query_fragment);
  ORT_UNUSED_PARAMETER(key_fragment);
#endif
}

__device__ __forceinline__ void SetValidationError(int32_t* error_flag,
                                                   DynamicSparseAttentionValidationError error) {
  atomicCAS(error_flag, kDynamicSparseAttentionValidationOk, static_cast<int32_t>(error));
}

__global__ void ValidateInputsKernel(const int32_t* selected_indices,
                                     const int32_t* selected_counts,
                                     const int32_t* seqlens_k,
                                     const int64_t* position_ids,
                                     int batch_size,
                                     int sequence_length,
                                     int max_selected,
                                     int maximum_total_length,
                                     int main_capacity,
                                     int auxiliary_sequence_length,
                                     int rotary_max_position,
                                     bool do_rotary,
                                     bool use_auxiliary,
                                     uint32_t* validation_hash_table,
                                     size_t validation_hash_entries,
                                     bool use_shared_hash_table,
                                     int32_t* error_flag) {
  const int row = static_cast<int>(blockIdx.x);
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  if (b >= batch_size) {
    return;
  }

  extern __shared__ uint32_t shared_hash_table[];
  if (use_shared_hash_table) {
    for (size_t i = threadIdx.x; i < validation_hash_entries; i += blockDim.x) {
      shared_hash_table[i] = std::numeric_limits<uint32_t>::max();
    }
  }
  __syncthreads();

  __shared__ int count;
  __shared__ int source_length;
  __shared__ int query_position;
  __shared__ int first_error_index;
  __shared__ bool row_is_valid;
  if (threadIdx.x == 0) {
    first_error_index = std::numeric_limits<int>::max();
    row_is_valid = false;
    const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
    if (total_length_64 < sequence_length ||
        total_length_64 > maximum_total_length ||
        total_length_64 > main_capacity) {
      SetValidationError(error_flag, kDynamicSparseAttentionInvalidSequenceLength);
    } else {
      const int total_length = static_cast<int>(total_length_64);
      query_position = total_length - sequence_length + s;
      if (do_rotary) {
        const int64_t position = position_ids == nullptr
                                     ? static_cast<int64_t>(query_position)
                                     : position_ids[row];
        if (position < 0 || position >= rotary_max_position) {
          SetValidationError(error_flag, kDynamicSparseAttentionInvalidPosition);
        } else {
          row_is_valid = true;
        }
      } else {
        row_is_valid = true;
      }

      count = selected_counts[row];
      if (count < 0 || count > max_selected) {
        SetValidationError(error_flag, kDynamicSparseAttentionInvalidCount);
        row_is_valid = false;
      }
      source_length = use_auxiliary ? auxiliary_sequence_length : total_length;
    }
  }
  __syncthreads();
  if (!row_is_valid) {
    return;
  }

  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;
  uint32_t* row_hash_table =
      use_shared_hash_table
          ? shared_hash_table
          : validation_hash_table + static_cast<size_t>(row) * validation_hash_entries;
  for (int i = static_cast<int>(threadIdx.x); i < max_selected; i += static_cast<int>(blockDim.x)) {
    const int index = row_indices[i];
    DynamicSparseAttentionValidationError error = kDynamicSparseAttentionValidationOk;
    if (i >= count) {
      if (index != -1) {
        error = kDynamicSparseAttentionInvalidPadding;
      }
    } else if (index < 0) {
      error = kDynamicSparseAttentionNegativeIndex;
    } else if (index >= source_length) {
      error = kDynamicSparseAttentionIndexOutOfBounds;
    } else if (!use_auxiliary && index > query_position) {
      error = kDynamicSparseAttentionNonCausalIndex;
    } else {
      const uint32_t hash_mask = static_cast<uint32_t>(validation_hash_entries - 1);
      uint32_t slot = static_cast<uint32_t>(index) * 0x9E3779B9U & hash_mask;
      for (size_t probe = 0; probe < validation_hash_entries; ++probe) {
        const uint32_t previous = atomicCAS(
            row_hash_table + slot, std::numeric_limits<uint32_t>::max(), static_cast<uint32_t>(index));
        if (previous == std::numeric_limits<uint32_t>::max()) {
          break;
        }
        if (previous == static_cast<uint32_t>(index)) {
          error = kDynamicSparseAttentionDuplicateIndex;
          break;
        }
        slot = (slot + 1) & hash_mask;
      }
    }
    if (error != kDynamicSparseAttentionValidationOk) {
      atomicMin(&first_error_index, i);
    }
  }
  __syncthreads();
  if (first_error_index != std::numeric_limits<int>::max()) {
    if (threadIdx.x == 0) {
      const int i = first_error_index;
      const int index = row_indices[i];
      DynamicSparseAttentionValidationError error;
      if (i >= count) {
        error = kDynamicSparseAttentionInvalidPadding;
      } else if (index < 0) {
        error = kDynamicSparseAttentionNegativeIndex;
      } else if (index >= source_length) {
        error = kDynamicSparseAttentionIndexOutOfBounds;
      } else if (!use_auxiliary && index > query_position) {
        error = kDynamicSparseAttentionNonCausalIndex;
      } else {
        error = kDynamicSparseAttentionDuplicateIndex;
      }
      SetValidationError(error_flag, error);
    }
    return;
  }
}

template <typename T>
__device__ __forceinline__ T ApplyRotary(const T* head_values,
                                         int h,
                                         int position,
                                         const T* cos_cache,
                                         const T* sin_cache,
                                         int rotary_dim,
                                         int rotary_offset,
                                         bool interleaved) {
  const int relative_h = h - rotary_offset;
  if (relative_h < 0 || relative_h >= rotary_dim) {
    return head_values[h];
  }

  const int half_rotary_dim = rotary_dim / 2;
  int cache_index;
  int partner;
  float sign;
  if (interleaved) {
    cache_index = relative_h / 2;
    partner = (relative_h % 2 == 0) ? relative_h + 1 : relative_h - 1;
    sign = relative_h % 2 == 0 ? -1.0f : 1.0f;
  } else {
    cache_index = relative_h % half_rotary_dim;
    partner = (relative_h + half_rotary_dim) % rotary_dim;
    sign = relative_h < half_rotary_dim ? -1.0f : 1.0f;
  }

  const int64_t cache_offset = static_cast<int64_t>(position) * half_rotary_dim + cache_index;
  const float value = ToFloat(head_values[h]) * ToFloat(cos_cache[cache_offset]) +
                      sign * ToFloat(head_values[rotary_offset + partner]) *
                          ToFloat(sin_cache[cache_offset]);
  return FromFloat<T>(value);
}

template <typename T>
__global__ void PrepareQueryKernel(const T* query,
                                   T* prepared_query,
                                   const int32_t* seqlens_k,
                                   const int64_t* position_ids,
                                   const T* cos_cache,
                                   const T* sin_cache,
                                   const T* norm_weight,
                                   int block_count,
                                   int sequence_length,
                                   int num_heads,
                                   int head_size,
                                   int input_stride,
                                   int rotary_dim,
                                   int rotary_max_position,
                                   int rotary_offset,
                                   bool rotary_interleaved,
                                   float epsilon) {
  const int block = static_cast<int>(blockIdx.x);
  if (block >= block_count) {
    return;
  }
  const int head = block % num_heads;
  const int row = block / num_heads;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int h = static_cast<int>(threadIdx.x);

  extern __shared__ unsigned char shared_bytes[];
  float* reduction = reinterpret_cast<float*>(shared_bytes);
  T* head_values = reinterpret_cast<T*>(reduction + blockDim.x);

  float value = 0.0f;
  if (h < head_size) {
    const int64_t input_offset = static_cast<int64_t>(row) * input_stride +
                                 static_cast<int64_t>(head) * head_size + h;
    value = ToFloat(query[input_offset]);
  }
  reduction[h] = norm_weight == nullptr ? 0.0f : value * value;
  __syncthreads();

  if (norm_weight != nullptr) {
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (h < static_cast<int>(stride)) {
        reduction[h] += reduction[h + stride];
      }
      __syncthreads();
    }
    if (h < head_size) {
      value *= rsqrtf(reduction[0] / static_cast<float>(head_size) + epsilon) *
               ToFloat(norm_weight[h]);
    }
  }
  if (h < head_size) {
    head_values[h] = FromFloat<T>(value);
  }
  __syncthreads();

  if (h < head_size) {
    T result = head_values[h];
    if (rotary_dim > 0) {
      const int64_t position = position_ids == nullptr
                                   ? static_cast<int64_t>(seqlens_k[b]) + 1 - sequence_length + s
                                   : position_ids[row];
      if (position >= 0 && position < rotary_max_position) {
        result = ApplyRotary(head_values, h, static_cast<int>(position), cos_cache, sin_cache,
                             rotary_dim, rotary_offset, rotary_interleaved);
      }
    }
    const int64_t output_offset =
        (static_cast<int64_t>(row) * num_heads + head) * head_size + h;
    prepared_query[output_offset] = result;
  }
}

template <typename T>
__global__ void AppendKvKernel(const T* query,
                               const T* key,
                               const T* value,
                               T* present_key,
                               T* present_value,
                               const int32_t* seqlens_k,
                               const int64_t* position_ids,
                               const T* cos_cache,
                               const T* sin_cache,
                               const T* norm_weight,
                               int block_count,
                               int sequence_length,
                               int kv_num_heads,
                               int head_size,
                               int query_hidden_size,
                               int kv_hidden_size,
                               int packed_stride,
                               int cache_capacity,
                               int rotary_dim,
                               int rotary_max_position,
                               int rotary_offset,
                               bool rotary_interleaved,
                               bool is_packed,
                               float epsilon) {
  const int block = static_cast<int>(blockIdx.x);
  if (block >= block_count) {
    return;
  }
  const int kv_head = block % kv_num_heads;
  const int row = block / kv_num_heads;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int64_t destination_sequence_64 =
      static_cast<int64_t>(seqlens_k[b]) + 1 - sequence_length + s;
  if (destination_sequence_64 < 0 || destination_sequence_64 >= cache_capacity) {
    return;
  }
  const int destination_sequence = static_cast<int>(destination_sequence_64);
  const int h = static_cast<int>(threadIdx.x);

  extern __shared__ unsigned char shared_bytes[];
  float* reduction = reinterpret_cast<float*>(shared_bytes);
  T* head_values = reinterpret_cast<T*>(reduction + blockDim.x);

  const int64_t row_start = static_cast<int64_t>(row) *
                            (is_packed ? packed_stride : kv_hidden_size);
  const int64_t key_offset = row_start +
                             (is_packed ? query_hidden_size : 0) +
                             static_cast<int64_t>(kv_head) * head_size + h;
  const int64_t value_offset = row_start +
                               (is_packed ? query_hidden_size + kv_hidden_size : 0) +
                               static_cast<int64_t>(kv_head) * head_size + h;
  float key_value = h < head_size ? ToFloat((is_packed ? query : key)[key_offset]) : 0.0f;
  reduction[h] = norm_weight == nullptr ? 0.0f : key_value * key_value;
  __syncthreads();

  if (norm_weight != nullptr) {
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (h < static_cast<int>(stride)) {
        reduction[h] += reduction[h + stride];
      }
      __syncthreads();
    }
    if (h < head_size) {
      key_value *= rsqrtf(reduction[0] / static_cast<float>(head_size) + epsilon) *
                   ToFloat(norm_weight[h]);
    }
  }
  if (h < head_size) {
    head_values[h] = FromFloat<T>(key_value);
  }
  __syncthreads();

  if (h < head_size) {
    T result = head_values[h];
    if (rotary_dim > 0) {
      const int64_t position = position_ids == nullptr
                                   ? static_cast<int64_t>(destination_sequence)
                                   : position_ids[row];
      if (position >= 0 && position < rotary_max_position) {
        result = ApplyRotary(head_values, h, static_cast<int>(position), cos_cache, sin_cache,
                             rotary_dim, rotary_offset, rotary_interleaved);
      }
    }
    const int64_t cache_offset =
        ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * cache_capacity +
         destination_sequence) *
            head_size +
        h;
    present_key[cache_offset] = result;
    present_value[cache_offset] = (is_packed ? query : value)[value_offset];
  }
}

template <typename T>
__global__ void FusedDynamicSparseAttentionKernel(const T* query,
                                                  const T* main_key,
                                                  const T* main_value,
                                                  const T* auxiliary_key,
                                                  const T* auxiliary_value,
                                                  const int32_t* selected_indices,
                                                  const int32_t* selected_counts,
                                                  const int32_t* seqlens_k,
                                                  const T* head_sink,
                                                  T* output,
                                                  int block_count,
                                                  int sequence_length,
                                                  int num_heads,
                                                  int kv_num_heads,
                                                  int head_size,
                                                  int main_capacity,
                                                  int auxiliary_sequence_length,
                                                  int max_selected,
                                                  int local_window_size,
                                                  int candidate_capacity,
                                                  float scale,
                                                  bool local_plus_selected,
                                                  bool selected_from_auxiliary,
                                                  bool use_smooth_softmax) {
  const int block = static_cast<int>(blockIdx.x);
  if (block >= block_count) {
    return;
  }

  const int head = block % num_heads;
  const int row = block / num_heads;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int h = static_cast<int>(threadIdx.x);
  const int lane = h & 31;
  const int warp = h >> 5;
  const int warp_count = static_cast<int>(blockDim.x) >> 5;
  const int kv_head = head / (num_heads / kv_num_heads);
  const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
  const bool valid_length = total_length_64 >= sequence_length && total_length_64 <= main_capacity;
  const int total_length = valid_length ? static_cast<int>(total_length_64) : 0;
  const int query_position = valid_length ? total_length - sequence_length + s : -1;
  const T* query_head = query + (static_cast<int64_t>(row) * num_heads + head) * head_size;

  int local_start = 0;
  int local_count = 0;
  if (local_plus_selected) {
    local_start = max(0, query_position - local_window_size + 1);
    const int local_end = min(query_position, min(total_length, main_capacity) - 1);
    local_count = max(0, local_end - local_start + 1);
  }
  int selected_count = selected_counts[row];
  selected_count = max(0, min(selected_count, max_selected));
  const int candidate_count = local_count + selected_count;
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;

  extern __shared__ float shared[];
  float* logits = shared;
  float* valid_candidates = logits + candidate_capacity;
  float* reduction = valid_candidates + candidate_capacity;
  T* shared_query = reinterpret_cast<T*>(reduction + blockDim.x);
  if (h < head_size) {
    shared_query[h] = query_head[h];
  }
  __syncthreads();

  for (int candidate = warp; candidate < candidate_count; candidate += warp_count) {
    int index;
    const T* key_head;
    bool valid = true;
    if (candidate < local_count) {
      index = local_start + candidate;
      const int64_t offset =
          ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
      key_head = main_key + offset;
    } else {
      index = row_indices[candidate - local_count];
      if (selected_from_auxiliary) {
        valid = index >= 0 && index < auxiliary_sequence_length;
        if (valid) {
          const int64_t offset =
              ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * auxiliary_sequence_length + index) * head_size;
          key_head = auxiliary_key + offset;
        }
      } else {
        valid = index >= 0 && index < total_length && index <= query_position && index < main_capacity;
        if (local_plus_selected && index >= local_start && index <= query_position) {
          valid = false;
        }
        if (valid) {
          const int64_t offset =
              ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
          key_head = main_key + offset;
        }
      }
    }

    float dot = 0.0f;
    if (valid) {
      for (int d = lane; d < head_size; d += 32) {
        dot += ToFloat(shared_query[d]) * ToFloat(key_head[d]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
      }
    }
    if (lane == 0) {
      logits[candidate] = valid ? dot * scale : 0.0f;
      valid_candidates[candidate] = valid ? 1.0f : 0.0f;
    }
  }
  __syncthreads();

  float thread_max = use_smooth_softmax && h == 0
                         ? (head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]))
                         : -FLT_MAX;
  for (int candidate = h; candidate < candidate_count; candidate += static_cast<int>(blockDim.x)) {
    if (valid_candidates[candidate] != 0.0f) {
      thread_max = fmaxf(thread_max, logits[candidate]);
    }
  }
  reduction[h] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (h < static_cast<int>(stride)) {
      reduction[h] = fmaxf(reduction[h], reduction[h + stride]);
    }
    __syncthreads();
  }
  const float max_logit = reduction[0];
  __syncthreads();

  float thread_sum = 0.0f;
  if (use_smooth_softmax && h == 0) {
    const float sink = head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]);
    thread_sum = expf(sink - max_logit);
  }
  for (int candidate = h; candidate < candidate_count; candidate += static_cast<int>(blockDim.x)) {
    const float logit = logits[candidate];
    const float weight = valid_candidates[candidate] != 0.0f ? expf(logit - max_logit) : 0.0f;
    logits[candidate] = weight;
    thread_sum += weight;
  }
  reduction[h] = thread_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (h < static_cast<int>(stride)) {
      reduction[h] += reduction[h + stride];
    }
    __syncthreads();
  }
  const float denominator = reduction[0];
  __syncthreads();

  if (h < head_size) {
    float accumulator = 0.0f;
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      const float weight = logits[candidate];
      if (valid_candidates[candidate] == 0.0f) {
        continue;
      }

      int index;
      const T* value_head;
      if (candidate < local_count) {
        index = local_start + candidate;
        const int64_t offset =
            ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
        value_head = main_value + offset;
      } else {
        index = row_indices[candidate - local_count];
        if (selected_from_auxiliary) {
          const int64_t offset =
              ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * auxiliary_sequence_length + index) * head_size;
          value_head = auxiliary_value + offset;
        } else {
          const int64_t offset =
              ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
          value_head = main_value + offset;
        }
      }
      accumulator += weight * ToFloat(value_head[h]);
    }

    const int64_t output_offset =
        (static_cast<int64_t>(row) * num_heads + head) * head_size + h;
    output[output_offset] =
        denominator == 0.0f ? FromFloat<T>(0.0f) : FromFloat<T>(accumulator / denominator);
  }
}

template <typename T, int kQueryHeadsPerBlock>
__global__ void GroupedPrefillDynamicSparseAttentionKernel(
    const T* query,
    const T* main_key,
    const T* main_value,
    const T* auxiliary_key,
    const T* auxiliary_value,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const T* head_sink,
    T* output,
    int block_count,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int main_capacity,
    int auxiliary_sequence_length,
    int max_selected,
    int local_window_size,
    float scale,
    bool local_plus_selected,
    bool selected_from_auxiliary,
    bool use_smooth_softmax) {
  const int block = static_cast<int>(blockIdx.x);
  if (block >= block_count) {
    return;
  }

  const int query_heads_per_kv = num_heads / kv_num_heads;
  const int head_tiles = (query_heads_per_kv + kQueryHeadsPerBlock - 1) / kQueryHeadsPerBlock;
  const int row = block / (kv_num_heads * head_tiles);
  const int kv_tile = block - row * kv_num_heads * head_tiles;
  const int kv_head = kv_tile / head_tiles;
  const int head_tile = kv_tile - kv_head * head_tiles;
  const int group_head = head_tile * kQueryHeadsPerBlock + static_cast<int>(threadIdx.x) / 32;
  const bool active_head = group_head < query_heads_per_kv;
  const int head = kv_head * query_heads_per_kv + group_head;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
  const bool valid_length = total_length_64 >= sequence_length && total_length_64 <= main_capacity;
  const int total_length = valid_length ? static_cast<int>(total_length_64) : 0;
  const int query_position = valid_length ? total_length - sequence_length + s : -1;

  int local_start = 0;
  int local_count = 0;
  if (local_plus_selected) {
    local_start = max(0, query_position - local_window_size + 1);
    const int local_end = min(query_position, min(total_length, main_capacity) - 1);
    local_count = max(0, local_end - local_start + 1);
  }
  const int selected_count = max(0, min(selected_counts[row], max_selected));
  const int candidate_count = local_count + selected_count;
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;

  extern __shared__ unsigned char shared_bytes[];
  T* shared_key = reinterpret_cast<T*>(shared_bytes);
  T* shared_value = shared_key + kGroupedPrefillCandidatesPerTile * head_size;
  __shared__ bool candidate_valid[kGroupedPrefillCandidatesPerTile];
  __shared__ half mma_query[kGroupedPrefillQueryHeadsPerBlock * 256];
  __shared__ float mma_scores[kGroupedPrefillQueryHeadsPerBlock * kGroupedPrefillCandidatesPerTile];

  constexpr int kValuesPerLane = 8;
  float query_values[kValuesPerLane]{};
  float accumulator[kValuesPerLane]{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same_v<T, half>) {
    const int group_head_base = head_tile * kQueryHeadsPerBlock;
    for (int index = static_cast<int>(threadIdx.x);
         index < kGroupedPrefillQueryHeadsPerBlock * head_size;
         index += static_cast<int>(blockDim.x)) {
      const int local_head = index / head_size;
      const int head_offset = index - local_head * head_size;
      const int staged_group_head = group_head_base + local_head;
      mma_query[index] = staged_group_head < query_heads_per_kv
                             ? query[(static_cast<int64_t>(row) * num_heads +
                                      kv_head * query_heads_per_kv + staged_group_head) *
                                         head_size +
                                     head_offset]
                             : half{0.0f};
    }
    __syncthreads();
  } else
#endif
      if (active_head) {
    const T* query_head = query + (static_cast<int64_t>(row) * num_heads + head) * head_size;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      query_values[i] = ToFloat(query_head[lane + i * 32]);
    }
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  if (active_head && use_smooth_softmax) {
    running_max = head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]);
    running_sum = 1.0f;
  }

  for (int candidate_start = 0; candidate_start < candidate_count;
       candidate_start += kGroupedPrefillCandidatesPerTile) {
    const int tile_count = min(kGroupedPrefillCandidatesPerTile, candidate_count - candidate_start);
    const int load_warp = static_cast<int>(threadIdx.x) / 32;
    const int load_lane = static_cast<int>(threadIdx.x) & 31;
    for (int load_candidate = load_warp; load_candidate < tile_count;
         load_candidate += kQueryHeadsPerBlock) {
      int candidate_index = 0;
      int candidate_is_auxiliary = 0;
      int is_valid = 0;
      if (load_lane == 0) {
        const int candidate = candidate_start + load_candidate;
        is_valid = valid_length;
        if (candidate < local_count) {
          candidate_index = local_start + candidate;
        } else {
          candidate_index = row_indices[candidate - local_count];
          candidate_is_auxiliary = selected_from_auxiliary;
          if (selected_from_auxiliary) {
            is_valid = is_valid && candidate_index >= 0 &&
                       candidate_index < auxiliary_sequence_length;
          } else {
            is_valid = is_valid && candidate_index >= 0 && candidate_index < total_length &&
                       candidate_index <= query_position && candidate_index < main_capacity;
            if (local_plus_selected && candidate_index >= local_start &&
                candidate_index <= query_position) {
              is_valid = false;
            }
          }
        }
        candidate_valid[load_candidate] = is_valid;
      }
      candidate_index = __shfl_sync(0xffffffff, candidate_index, 0);
      candidate_is_auxiliary = __shfl_sync(0xffffffff, candidate_is_auxiliary, 0);
      is_valid = __shfl_sync(0xffffffff, is_valid, 0);

      const int shared_offset = load_candidate * head_size;
      if (is_valid) {
        const int64_t source_length = candidate_is_auxiliary
                                          ? auxiliary_sequence_length
                                          : main_capacity;
        const int64_t offset =
            ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * source_length +
             candidate_index) *
            head_size;
        const T* key_head = candidate_is_auxiliary
                                ? auxiliary_key + offset
                                : main_key + offset;
        const T* value_head = candidate_is_auxiliary
                                  ? auxiliary_value + offset
                                  : main_value + offset;
        reinterpret_cast<uint4*>(shared_key + shared_offset)[load_lane] =
            reinterpret_cast<const uint4*>(key_head)[load_lane];
        reinterpret_cast<uint4*>(shared_value + shared_offset)[load_lane] =
            reinterpret_cast<const uint4*>(value_head)[load_lane];
      }
    }
    __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (std::is_same_v<T, half>) {
      if (threadIdx.x < 32) {
        const int mma_lane = static_cast<int>(threadIdx.x);
        const int row_pair = mma_lane >> 2;
        const int k_pair = (mma_lane & 3) * 2;
        float score_fragment[4]{};
        for (int k_start = 0; k_start < head_size; k_start += 16) {
          const int first_k = k_start + k_pair;
          const int second_k = first_k + 8;
          uint32_t query_fragment[4]{};
          if (row_pair < kGroupedPrefillQueryHeadsPerBlock) {
            query_fragment[0] = *reinterpret_cast<const uint32_t*>(
                mma_query + row_pair * head_size + first_k);
            query_fragment[2] = *reinterpret_cast<const uint32_t*>(
                mma_query + row_pair * head_size + second_k);
          }
          const uint32_t key_fragment[2]{
              *reinterpret_cast<const uint32_t*>(
                  shared_key + row_pair * head_size + first_k),
              *reinterpret_cast<const uint32_t*>(
                  shared_key + row_pair * head_size + second_k)};
          MmaM16N8K16(score_fragment, query_fragment, key_fragment);
        }
        const int score_column = (mma_lane & 3) * 2;
        if (row_pair < kGroupedPrefillQueryHeadsPerBlock) {
          mma_scores[row_pair * kGroupedPrefillCandidatesPerTile + score_column] = score_fragment[0];
          mma_scores[row_pair * kGroupedPrefillCandidatesPerTile + score_column + 1] = score_fragment[1];
        }
      }
      __syncthreads();
    }
#endif

    for (int tile_candidate = 0; tile_candidate < tile_count; ++tile_candidate) {
      if (active_head && candidate_valid[tile_candidate]) {
        const int shared_offset = tile_candidate * head_size;
        float score;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half>) {
          score = mma_scores[static_cast<int>(threadIdx.x) / 32 *
                                 kGroupedPrefillCandidatesPerTile +
                             tile_candidate] *
                  scale;
        } else
#endif
        {
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < kValuesPerLane; ++i) {
            dot += query_values[i] * ToFloat(shared_key[shared_offset + lane + i * 32]);
          }
          for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
          }
          score = __shfl_sync(0xffffffff, dot, 0) * scale;
        }
        const float next_max = fmaxf(running_max, score);
        const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float weight = expf(score - next_max);
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          accumulator[i] = accumulator[i] * old_scale +
                           weight * ToFloat(shared_value[shared_offset + lane + i * 32]);
        }
        running_max = next_max;
        running_sum = running_sum * old_scale + weight;
      }
    }
    __syncthreads();
  }

  if (active_head) {
    T* output_head = output + (static_cast<int64_t>(row) * num_heads + head) * head_size;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      output_head[lane + i * 32] = running_sum == 0.0f
                                       ? FromFloat<T>(0.0f)
                                       : FromFloat<T>(accumulator[i] / running_sum);
    }
  }
}

template <typename T, int kQueryHeadsPerBlock, bool kWarpLocalMetadata>
__global__ void GroupedDecodeDynamicSparseAttentionKernel(
    const T* query,
    const T* main_key,
    const T* main_value,
    const T* auxiliary_key,
    const T* auxiliary_value,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const T* head_sink,
    float* partial_max,
    float* partial_sum,
    float* partial_output,
    int grouped_block_count,
    int split_count,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int main_capacity,
    int auxiliary_sequence_length,
    int max_selected,
    int local_window_size,
    float scale,
    bool local_plus_selected,
    bool selected_from_auxiliary,
    bool use_smooth_softmax) {
  const int block = static_cast<int>(blockIdx.x);
  if (block >= grouped_block_count) {
    return;
  }

  const int split = block % split_count;
  const int grouped_block = block / split_count;
  const int query_heads_per_kv = num_heads / kv_num_heads;
  const int head_tiles = (query_heads_per_kv + kQueryHeadsPerBlock - 1) / kQueryHeadsPerBlock;
  const int b = grouped_block / (kv_num_heads * head_tiles);
  const int kv_tile = grouped_block - b * kv_num_heads * head_tiles;
  const int kv_head = kv_tile / head_tiles;
  const int head_tile = kv_tile - kv_head * head_tiles;
  const int group_head = head_tile * kQueryHeadsPerBlock + static_cast<int>(threadIdx.x) / 32;
  const bool active_head = group_head < query_heads_per_kv;
  const int head = kv_head * query_heads_per_kv + group_head;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int total_length = seqlens_k[b] + 1;
  const int query_position = total_length - 1;

  int local_start = 0;
  int local_count = 0;
  if (local_plus_selected) {
    local_start = max(0, query_position - local_window_size + 1);
    const int local_end = min(query_position, min(total_length, main_capacity) - 1);
    local_count = max(0, local_end - local_start + 1);
  }
  const int selected_count = max(0, min(selected_counts[b], max_selected));
  const int candidate_count = local_count + selected_count;
  const int candidate_begin = split * kGroupedDecodeCandidatesPerSplit;
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(b) * max_selected;

  extern __shared__ unsigned char shared_bytes[];
  T* shared_key = reinterpret_cast<T*>(shared_bytes);
  T* shared_value = shared_key + kGroupedPrefillCandidatesPerTile * head_size;
  __shared__ int candidate_indices[kGroupedPrefillCandidatesPerTile];
  __shared__ bool candidate_valid[kGroupedPrefillCandidatesPerTile];
  __shared__ bool candidate_sources[kGroupedPrefillCandidatesPerTile];
  __shared__ half mma_query[kGroupedDecodeQueryHeadsPerBlock * 256];
  __shared__ float mma_scores[kGroupedDecodeQueryHeadsPerBlock * kGroupedPrefillCandidatesPerTile];

  constexpr int kValuesPerLane = 8;
  float query_values[kValuesPerLane]{};
  float accumulator[kValuesPerLane]{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same_v<T, half>) {
    const int group_head_base = head_tile * kQueryHeadsPerBlock;
    for (int index = static_cast<int>(threadIdx.x);
         index < kGroupedDecodeQueryHeadsPerBlock * head_size;
         index += static_cast<int>(blockDim.x)) {
      const int local_head = index / head_size;
      const int head_offset = index - local_head * head_size;
      const int staged_group_head = group_head_base + local_head;
      mma_query[index] = staged_group_head < query_heads_per_kv
                             ? query[(static_cast<int64_t>(b) * num_heads +
                                      kv_head * query_heads_per_kv + staged_group_head) *
                                         head_size +
                                     head_offset]
                             : half{0.0f};
    }
    __syncthreads();
  } else
#endif
      if (active_head) {
    const T* query_head = query + (static_cast<int64_t>(b) * num_heads + head) * head_size;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      query_values[i] = ToFloat(query_head[lane + i * 32]);
    }
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  if (active_head && split == 0 && use_smooth_softmax) {
    running_max = head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]);
    running_sum = 1.0f;
  }

  for (int candidate_start = candidate_begin; candidate_start < candidate_count;) {
    const int tile_count = min(kGroupedPrefillCandidatesPerTile, candidate_count - candidate_start);
    const int load_candidate = static_cast<int>(threadIdx.x) / 32;
    const int load_lane = static_cast<int>(threadIdx.x) & 31;
    int candidate_index = 0;
    int candidate_is_auxiliary = 0;
    int is_valid = 0;
    if constexpr (!kWarpLocalMetadata) {
      if (threadIdx.x < tile_count) {
        const int tile_candidate = static_cast<int>(threadIdx.x);
        const int candidate = candidate_start + tile_candidate;
        candidate_valid[tile_candidate] = total_length > 0 && total_length <= main_capacity;
        candidate_sources[tile_candidate] = false;
        if (candidate < local_count) {
          candidate_indices[tile_candidate] = local_start + candidate;
        } else {
          candidate_indices[tile_candidate] = row_indices[candidate - local_count];
          candidate_sources[tile_candidate] = selected_from_auxiliary;
          if (selected_from_auxiliary) {
            candidate_valid[tile_candidate] =
                candidate_valid[tile_candidate] && candidate_indices[tile_candidate] >= 0 &&
                candidate_indices[tile_candidate] < auxiliary_sequence_length;
          } else {
            candidate_valid[tile_candidate] =
                candidate_valid[tile_candidate] && candidate_indices[tile_candidate] >= 0 &&
                candidate_indices[tile_candidate] < total_length &&
                candidate_indices[tile_candidate] <= query_position;
            if (local_plus_selected && candidate_indices[tile_candidate] >= local_start) {
              candidate_valid[tile_candidate] = false;
            }
          }
        }
      }
      __syncthreads();
    }
    if (load_candidate < tile_count) {
      if constexpr (kWarpLocalMetadata) {
        if (load_lane == 0) {
          const int candidate = candidate_start + load_candidate;
          is_valid = total_length > 0 && total_length <= main_capacity;
          if (candidate < local_count) {
            candidate_index = local_start + candidate;
          } else {
            candidate_index = row_indices[candidate - local_count];
            candidate_is_auxiliary = selected_from_auxiliary;
            if (selected_from_auxiliary) {
              is_valid = is_valid && candidate_index >= 0 &&
                         candidate_index < auxiliary_sequence_length;
            } else {
              is_valid = is_valid && candidate_index >= 0 && candidate_index < total_length &&
                         candidate_index <= query_position;
              if (local_plus_selected && candidate_index >= local_start) {
                is_valid = false;
              }
            }
          }
          candidate_valid[load_candidate] = is_valid;
        }
        candidate_index = __shfl_sync(0xffffffff, candidate_index, 0);
        candidate_is_auxiliary = __shfl_sync(0xffffffff, candidate_is_auxiliary, 0);
        is_valid = __shfl_sync(0xffffffff, is_valid, 0);
      } else {
        candidate_index = candidate_indices[load_candidate];
        candidate_is_auxiliary = candidate_sources[load_candidate];
        is_valid = candidate_valid[load_candidate];
      }

      const int shared_offset = load_candidate * head_size;
      if (is_valid) {
        const int64_t source_length = candidate_is_auxiliary
                                          ? auxiliary_sequence_length
                                          : main_capacity;
        const int64_t offset =
            ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * source_length +
             candidate_index) *
            head_size;
        const T* key_head = candidate_is_auxiliary
                                ? auxiliary_key + offset
                                : main_key + offset;
        const T* value_head = candidate_is_auxiliary
                                  ? auxiliary_value + offset
                                  : main_value + offset;
        reinterpret_cast<uint4*>(shared_key + shared_offset)[load_lane] =
            reinterpret_cast<const uint4*>(key_head)[load_lane];
        reinterpret_cast<uint4*>(shared_value + shared_offset)[load_lane] =
            reinterpret_cast<const uint4*>(value_head)[load_lane];
      }
    }
    __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (std::is_same_v<T, half>) {
      if (threadIdx.x < 32) {
        const int mma_lane = static_cast<int>(threadIdx.x);
        const int row_pair = mma_lane >> 2;
        const int k_pair = (mma_lane & 3) * 2;
        float score_fragment[4]{};
        for (int k_start = 0; k_start < head_size; k_start += 16) {
          const int first_k = k_start + k_pair;
          const int second_k = first_k + 8;
          const uint32_t query_fragment[4]{
              *reinterpret_cast<const uint32_t*>(
                  mma_query + row_pair * head_size + first_k),
              0,
              *reinterpret_cast<const uint32_t*>(
                  mma_query + row_pair * head_size + second_k),
              0};
          const uint32_t key_fragment[2]{
              *reinterpret_cast<const uint32_t*>(
                  shared_key + row_pair * head_size + first_k),
              *reinterpret_cast<const uint32_t*>(
                  shared_key + row_pair * head_size + second_k)};
          MmaM16N8K16(score_fragment, query_fragment, key_fragment);
        }
        const int score_column = (mma_lane & 3) * 2;
        mma_scores[row_pair * kGroupedPrefillCandidatesPerTile + score_column] = score_fragment[0];
        mma_scores[row_pair * kGroupedPrefillCandidatesPerTile + score_column + 1] = score_fragment[1];
      }
      __syncthreads();
    }
#endif

    for (int tile_candidate = 0; tile_candidate < tile_count; ++tile_candidate) {
      if (active_head && candidate_valid[tile_candidate]) {
        const int shared_offset = tile_candidate * head_size;
        float score;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half>) {
          score = mma_scores[static_cast<int>(threadIdx.x) / 32 *
                                 kGroupedPrefillCandidatesPerTile +
                             tile_candidate] *
                  scale;
        } else
#endif
        {
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < kValuesPerLane; ++i) {
            dot += query_values[i] * ToFloat(shared_key[shared_offset + lane + i * 32]);
          }
          for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
          }
          score = __shfl_sync(0xffffffff, dot, 0) * scale;
        }
        const float next_max = fmaxf(running_max, score);
        const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float weight = expf(score - next_max);
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          accumulator[i] = accumulator[i] * old_scale +
                           weight * ToFloat(shared_value[shared_offset + lane + i * 32]);
        }
        running_max = next_max;
        running_sum = running_sum * old_scale + weight;
      }
    }
    __syncthreads();

    candidate_start += kGroupedPrefillCandidatesPerTile;
    if (candidate_start % kGroupedDecodeCandidatesPerSplit == 0) {
      candidate_start += (split_count - 1) * kGroupedDecodeCandidatesPerSplit;
    }
  }

  if (active_head) {
    const int partial_block = (b * num_heads + head) * split_count + split;
    if (lane == 0) {
      partial_max[partial_block] = running_max;
      partial_sum[partial_block] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      partial_output[static_cast<int64_t>(partial_block) * head_size + lane + i * 32] = accumulator[i];
    }
  }
}

template <typename T>
__global__ void SplitDynamicSparseAttentionKernel(const T* query,
                                                  const T* main_key,
                                                  const T* main_value,
                                                  const T* auxiliary_key,
                                                  const T* auxiliary_value,
                                                  const int32_t* selected_indices,
                                                  const int32_t* selected_counts,
                                                  const int32_t* seqlens_k,
                                                  const T* head_sink,
                                                  float* partial_max,
                                                  float* partial_sum,
                                                  float* partial_output,
                                                  T* output,
                                                  int partial_block_count,
                                                  int split_count,
                                                  int sequence_length,
                                                  int num_heads,
                                                  int kv_num_heads,
                                                  int head_size,
                                                  int main_capacity,
                                                  int auxiliary_sequence_length,
                                                  int max_selected,
                                                  int local_window_size,
                                                  float scale,
                                                  bool local_plus_selected,
                                                  bool selected_from_auxiliary,
                                                  bool use_smooth_softmax) {
  const int partial_block = static_cast<int>(blockIdx.x);
  if (partial_block >= partial_block_count) {
    return;
  }

  const int split = partial_block % split_count;
  const int query_block = partial_block / split_count;
  const int head = query_block % num_heads;
  const int row = query_block / num_heads;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int h = static_cast<int>(threadIdx.x);
  const int lane = h & 31;
  const int warp = h >> 5;
  const int warp_count = static_cast<int>(blockDim.x) >> 5;
  const int kv_head = head / (num_heads / kv_num_heads);
  const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
  const bool valid_length = total_length_64 >= sequence_length && total_length_64 <= main_capacity;
  const int total_length = valid_length ? static_cast<int>(total_length_64) : 0;
  const int query_position = valid_length ? total_length - sequence_length + s : -1;
  const T* query_head = query + (static_cast<int64_t>(row) * num_heads + head) * head_size;

  int local_start = 0;
  int local_count = 0;
  if (local_plus_selected) {
    local_start = max(0, query_position - local_window_size + 1);
    const int local_end = min(query_position, min(total_length, main_capacity) - 1);
    local_count = max(0, local_end - local_start + 1);
  }
  const int selected_count = max(0, min(selected_counts[row], max_selected));
  const int candidate_count = local_count + selected_count;
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;

  extern __shared__ float shared[];
  float* logits = shared;
  float* valid_candidates = logits + kSplitCandidateSize;
  float* reduction = valid_candidates + kSplitCandidateSize;
  T* shared_query = reinterpret_cast<T*>(reduction + blockDim.x);
  if (h < head_size) {
    shared_query[h] = query_head[h];
  }
  __syncthreads();

  float running_max = use_smooth_softmax && split == 0
                          ? (head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]))
                          : -FLT_MAX;
  float running_sum = use_smooth_softmax && split == 0 ? 1.0f : 0.0f;
  float running_accumulator = 0.0f;
  const int candidate_stride = split_count * kSplitCandidateSize;
  for (int candidate_start = split * kSplitCandidateSize;
       candidate_start < candidate_count;
       candidate_start += candidate_stride) {
    const int split_candidate_count =
        min(kSplitCandidateSize, candidate_count - candidate_start);

    for (int split_candidate = warp; split_candidate < split_candidate_count;
         split_candidate += warp_count) {
      const int candidate = candidate_start + split_candidate;
      int index;
      const T* key_head = nullptr;
      bool valid = true;
      if (candidate < local_count) {
        index = local_start + candidate;
        const int64_t offset =
            ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
        key_head = main_key + offset;
      } else {
        index = row_indices[candidate - local_count];
        if (selected_from_auxiliary) {
          valid = index >= 0 && index < auxiliary_sequence_length;
          if (valid) {
            const int64_t offset =
                ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * auxiliary_sequence_length + index) * head_size;
            key_head = auxiliary_key + offset;
          }
        } else {
          valid = index >= 0 && index < total_length && index <= query_position && index < main_capacity;
          if (local_plus_selected && index >= local_start && index <= query_position) {
            valid = false;
          }
          if (valid) {
            const int64_t offset =
                ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
            key_head = main_key + offset;
          }
        }
      }

      float dot = 0.0f;
      if (valid) {
        for (int d = lane; d < head_size; d += 32) {
          dot += ToFloat(shared_query[d]) * ToFloat(key_head[d]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
          dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
      }
      if (lane == 0) {
        logits[split_candidate] = valid ? dot * scale : 0.0f;
        valid_candidates[split_candidate] = valid ? 1.0f : 0.0f;
      }
    }
    __syncthreads();

    float thread_max = -FLT_MAX;
    for (int candidate = h; candidate < split_candidate_count;
         candidate += static_cast<int>(blockDim.x)) {
      if (valid_candidates[candidate] != 0.0f) {
        thread_max = fmaxf(thread_max, logits[candidate]);
      }
    }
    reduction[h] = thread_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (h < static_cast<int>(stride)) {
        reduction[h] = fmaxf(reduction[h], reduction[h + stride]);
      }
      __syncthreads();
    }
    const float tile_max = reduction[0];
    const float next_max = fmaxf(running_max, tile_max);
    const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
    __syncthreads();

    float thread_sum = 0.0f;
    for (int candidate = h; candidate < split_candidate_count;
         candidate += static_cast<int>(blockDim.x)) {
      const float logit = logits[candidate];
      const float weight = valid_candidates[candidate] != 0.0f ? expf(logit - next_max) : 0.0f;
      logits[candidate] = weight;
      thread_sum += weight;
    }
    reduction[h] = thread_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (h < static_cast<int>(stride)) {
        reduction[h] += reduction[h + stride];
      }
      __syncthreads();
    }
    const float tile_sum = reduction[0];
    __syncthreads();

    if (h < head_size) {
      float tile_accumulator = 0.0f;
      for (int split_candidate = 0; split_candidate < split_candidate_count; ++split_candidate) {
        const float weight = logits[split_candidate];
        if (valid_candidates[split_candidate] == 0.0f) {
          continue;
        }

        const int candidate = candidate_start + split_candidate;
        int index;
        const T* value_head;
        if (candidate < local_count) {
          index = local_start + candidate;
          const int64_t offset =
              ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
          value_head = main_value + offset;
        } else {
          index = row_indices[candidate - local_count];
          if (selected_from_auxiliary) {
            const int64_t offset =
                ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * auxiliary_sequence_length + index) * head_size;
            value_head = auxiliary_value + offset;
          } else {
            const int64_t offset =
                ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
            value_head = main_value + offset;
          }
        }
        tile_accumulator += weight * ToFloat(value_head[h]);
      }
      running_accumulator =
          running_accumulator * old_scale + tile_accumulator;
    }

    running_max = next_max;
    running_sum = running_sum * old_scale + tile_sum;
    __syncthreads();
  }
  if (split_count == 1) {
    if (h < head_size) {
      output[static_cast<int64_t>(query_block) * head_size + h] =
          running_sum == 0.0f
              ? FromFloat<T>(0.0f)
              : FromFloat<T>(running_accumulator / running_sum);
    }
  } else {
    if (h == 0) {
      partial_max[partial_block] = running_max;
      partial_sum[partial_block] = running_sum;
    }
    if (h < head_size) {
      partial_output[static_cast<int64_t>(partial_block) * head_size + h] =
          running_accumulator;
    }
  }
}

template <typename T>
__global__ void MergeDynamicSparseAttentionKernel(const float* partial_max,
                                                  const float* partial_sum,
                                                  const float* partial_output,
                                                  T* output,
                                                  int query_block_count,
                                                  int split_count,
                                                  int head_size) {
  const int query_block = static_cast<int>(blockIdx.x);
  if (query_block >= query_block_count) {
    return;
  }
  const int h = static_cast<int>(threadIdx.x);
  const int partial_start = query_block * split_count;

  extern __shared__ float reduction[];
  float thread_max = -FLT_MAX;
  for (int split = h; split < split_count; split += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, partial_max[partial_start + split]);
  }
  reduction[h] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (h < static_cast<int>(stride)) {
      reduction[h] = fmaxf(reduction[h], reduction[h + stride]);
    }
    __syncthreads();
  }
  const float max_logit = reduction[0];
  __syncthreads();

  float thread_sum = 0.0f;
  for (int split = h; split < split_count; split += static_cast<int>(blockDim.x)) {
    const float split_max = partial_max[partial_start + split];
    thread_sum += partial_sum[partial_start + split] * expf(split_max - max_logit);
  }
  reduction[h] = thread_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (h < static_cast<int>(stride)) {
      reduction[h] += reduction[h + stride];
    }
    __syncthreads();
  }
  const float denominator = reduction[0];
  __syncthreads();

  if (h < head_size) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      const float split_max = partial_max[partial_start + split];
      const float merge_scale = expf(split_max - max_logit);
      const int64_t partial_offset =
          (static_cast<int64_t>(partial_start + split) * head_size) + h;
      accumulator += partial_output[partial_offset] * merge_scale;
    }
    output[static_cast<int64_t>(query_block) * head_size + h] =
        denominator == 0.0f ? FromFloat<T>(0.0f) : FromFloat<T>(accumulator / denominator);
  }
}

template <typename T>
__global__ void MergeGroupedDecodeDynamicSparseAttentionKernel(const float* partial_max,
                                                               const float* partial_sum,
                                                               const float* partial_output,
                                                               T* output,
                                                               int query_block_count,
                                                               int split_count,
                                                               int head_size) {
  const int query_block = static_cast<int>(blockIdx.x);
  if (query_block >= query_block_count) {
    return;
  }

  const int h = static_cast<int>(threadIdx.x);
  const int lane = h & 31;
  const int partial_start = query_block * split_count;
  __shared__ float max_logit;
  __shared__ float denominator;

  if (h < 32) {
    float value = lane < split_count ? partial_max[partial_start + lane] : -FLT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    if (lane == 0) {
      max_logit = value;
    }
  }
  __syncthreads();

  if (h < 32) {
    float value = 0.0f;
    if (lane < split_count && partial_max[partial_start + lane] != -FLT_MAX) {
      value = partial_sum[partial_start + lane] *
              expf(partial_max[partial_start + lane] - max_logit);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      denominator = value;
    }
  }
  __syncthreads();

  if (h < head_size) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      const float split_max = partial_max[partial_start + split];
      if (split_max != -FLT_MAX) {
        const int64_t partial_offset =
            (static_cast<int64_t>(partial_start + split) * head_size) + h;
        accumulator += partial_output[partial_offset] * expf(split_max - max_logit);
      }
    }
    output[static_cast<int64_t>(query_block) * head_size + h] =
        denominator == 0.0f ? FromFloat<T>(0.0f) : FromFloat<T>(accumulator / denominator);
  }
}

int GetThreadsPerBlock(int head_size) {
  int threads = 32;
  while (threads < head_size) {
    threads *= 2;
  }
  return threads;
}

Status CheckBlockCount(int64_t block_count, const char* kernel_name) {
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(),
                    "DynamicSparseAttention: ", kernel_name, " requires too many thread blocks.");
  return Status::OK();
}

int64_t GetCandidateCapacity(const DynamicSparseAttentionParameters& parameters) {
  return static_cast<int64_t>(parameters.max_selected) +
         (parameters.attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected
              ? std::min(parameters.local_window_size, parameters.cache_capacity)
              : 0);
}

int GetAttentionSplitCount(const DynamicSparseAttentionParameters& parameters) {
  const int64_t candidate_tiles =
      (GetCandidateCapacity(parameters) + kSplitCandidateSize - 1) /
      kSplitCandidateSize;
  const int64_t query_blocks =
      static_cast<int64_t>(parameters.batch_size) *
      parameters.sequence_length * parameters.num_heads;
  const int64_t occupancy_splits =
      std::max<int64_t>(1, (kTargetSplitBlocks + query_blocks - 1) / query_blocks);
  return static_cast<int>(std::min(candidate_tiles, occupancy_splits));
}

bool UseFusedAttention(const DynamicSparseAttentionParameters& parameters,
                       size_t element_size,
                       size_t max_shared_memory_per_block) {
  if (GetCandidateCapacity(parameters) > kSplitCandidateSize) {
    return false;
  }

  const int threads = GetThreadsPerBlock(parameters.head_size);
  const int fused_threads = threads < 128 ? 128 : threads;
  const size_t fused_shared_bytes =
      (2 * static_cast<size_t>(GetCandidateCapacity(parameters)) + static_cast<size_t>(fused_threads)) * sizeof(float) +
      static_cast<size_t>(parameters.head_size) * element_size;
  return fused_shared_bytes <= kMaxFusedSharedBytes &&
         fused_shared_bytes < max_shared_memory_per_block;
}

int GetGroupedHeadTiles(const DynamicSparseAttentionParameters& parameters,
                        int query_heads_per_block) {
  const int query_heads_per_kv = parameters.num_heads / parameters.kv_num_heads;
  return (query_heads_per_kv + query_heads_per_block - 1) / query_heads_per_block;
}

size_t GetGroupedAttentionSharedBytes(const DynamicSparseAttentionParameters& parameters,
                                      size_t element_size) {
  return 2 * kGroupedPrefillCandidatesPerTile *
         static_cast<size_t>(parameters.head_size) * element_size;
}

bool SupportsGroupedAttention(const DynamicSparseAttentionParameters& parameters,
                              size_t element_size,
                              size_t max_shared_memory_per_block) {
  return element_size == sizeof(half) &&
         parameters.head_size == 256 &&
         parameters.num_heads > parameters.kv_num_heads &&
         parameters.num_heads % parameters.kv_num_heads == 0 &&
         GetGroupedAttentionSharedBytes(parameters, element_size) < max_shared_memory_per_block;
}

bool UseGroupedPrefillAttention(const DynamicSparseAttentionParameters& parameters,
                                size_t element_size,
                                size_t max_shared_memory_per_block) {
  return parameters.sequence_length > 1 &&
         SupportsGroupedAttention(parameters, element_size, max_shared_memory_per_block);
}

bool UseGroupedDecodeAttention(const DynamicSparseAttentionParameters& parameters,
                               size_t element_size,
                               size_t max_shared_memory_per_block) {
  return parameters.sequence_length == 1 &&
         parameters.num_heads / parameters.kv_num_heads >= 4 &&
         GetCandidateCapacity(parameters) > kSplitCandidateSize &&
         SupportsGroupedAttention(parameters, element_size, max_shared_memory_per_block);
}

int GetGroupedDecodeSplitCount(const DynamicSparseAttentionParameters& parameters) {
  const int64_t candidate_splits =
      (GetCandidateCapacity(parameters) + kGroupedDecodeCandidatesPerSplit - 1) /
      kGroupedDecodeCandidatesPerSplit;
  const int64_t grouped_head_blocks =
      static_cast<int64_t>(parameters.batch_size) * parameters.kv_num_heads *
      GetGroupedHeadTiles(parameters, kGroupedDecodeQueryHeadsPerBlock);
  const int64_t occupancy_splits =
      std::max<int64_t>(1, (kTargetGroupedDecodeBlocks + grouped_head_blocks - 1) /
                               grouped_head_blocks);
  return static_cast<int>(std::min(candidate_splits, occupancy_splits));
}

size_t GetValidationHashEntries(const DynamicSparseAttentionParameters& parameters) {
  if (parameters.max_selected == 0) {
    return 0;
  }

  const size_t max_selected = static_cast<size_t>(parameters.max_selected);
  const size_t target_entries = max_selected + (max_selected + 1) / 2;
  size_t hash_entries = 1;
  while (hash_entries < target_entries) {
    hash_entries <<= 1;
  }
  return hash_entries;
}

bool UseSharedValidationHashTable(const DynamicSparseAttentionParameters& parameters,
                                  size_t max_shared_memory_per_block) {
  const size_t shared_bytes = GetValidationHashEntries(parameters) * sizeof(uint32_t);
  return shared_bytes <= kMaxValidationSharedBytes &&
         shared_bytes < max_shared_memory_per_block;
}

}  // namespace

size_t GetDynamicSparseAttentionValidationWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t max_shared_memory_per_block) {
  if (UseSharedValidationHashTable(parameters, max_shared_memory_per_block)) {
    return 0;
  }

  const size_t row_count =
      SafeInt<size_t>(parameters.batch_size) * parameters.sequence_length;
  return SafeInt<size_t>(row_count) * GetValidationHashEntries(parameters);
}

size_t GetDynamicSparseAttentionWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t element_size,
    size_t max_shared_memory_per_block) {
  if (UseGroupedPrefillAttention(parameters, element_size, max_shared_memory_per_block)) {
    return 0;
  }
  if (UseFusedAttention(parameters, element_size, max_shared_memory_per_block)) {
    return 0;
  }

  const size_t query_blocks =
      static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads;
  const size_t split_count =
      UseGroupedDecodeAttention(parameters, element_size, max_shared_memory_per_block)
          ? GetGroupedDecodeSplitCount(parameters)
          : GetAttentionSplitCount(parameters);
  if (split_count == 1) {
    return 0;
  }
  return SafeInt<size_t>(query_blocks) * split_count *
         (static_cast<size_t>(parameters.head_size) + 2);
}

Status ValidateDynamicSparseAttentionOnDevice(
    cudaStream_t stream,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    const DynamicSparseAttentionParameters& parameters,
    int32_t* error_flag,
    uint32_t* validation_bitmap,
    size_t max_shared_memory_per_block,
    bool copy_result_to_host) {
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(error_flag, 0, sizeof(int32_t), stream));
  const int64_t row_count =
      static_cast<int64_t>(parameters.batch_size) * parameters.sequence_length;
  const size_t validation_hash_entries = GetValidationHashEntries(parameters);
  const bool use_shared_hash_table =
      UseSharedValidationHashTable(parameters, max_shared_memory_per_block);
  const size_t validation_hash_elements =
      use_shared_hash_table ? 0 : static_cast<size_t>(row_count) * validation_hash_entries;
  if (validation_hash_elements > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        validation_bitmap, 0xff, validation_hash_elements * sizeof(uint32_t), stream));
  }
  ORT_RETURN_IF_ERROR(CheckBlockCount(row_count, "validation"));
  int validation_threads = 32;
  while (validation_threads < parameters.max_selected && validation_threads < 1024) {
    validation_threads <<= 1;
  }
  const size_t validation_shared_bytes =
      use_shared_hash_table ? validation_hash_entries * sizeof(uint32_t) : 0;
  ValidateInputsKernel<<<static_cast<int>(row_count), validation_threads,
                         validation_shared_bytes, stream>>>(
      selected_indices, selected_counts, seqlens_k, position_ids,
      parameters.batch_size, parameters.sequence_length, parameters.max_selected,
      parameters.total_sequence_length,
      parameters.cache_capacity, parameters.auxiliary_sequence_length,
      parameters.rotary_max_position, parameters.do_rotary,
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
      validation_bitmap, validation_hash_entries, use_shared_hash_table, error_flag);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  if (!copy_result_to_host) {
    return Status::OK();
  }

  int32_t host_error = kDynamicSparseAttentionValidationOk;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
      &host_error, error_flag, sizeof(host_error), cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  if (host_error == kDynamicSparseAttentionValidationOk) {
    return Status::OK();
  }

  const char* message = "selected inputs are invalid";
  switch (host_error) {
    case kDynamicSparseAttentionInvalidCount:
      message = "selected_counts values must be in [0, max_selected]";
      break;
    case kDynamicSparseAttentionInvalidPadding:
      message = "selected_indices entries after selected_counts must be -1";
      break;
    case kDynamicSparseAttentionNegativeIndex:
      message = "active selected_indices entries must be nonnegative";
      break;
    case kDynamicSparseAttentionIndexOutOfBounds:
      message = "selected_indices contains an out-of-bounds index";
      break;
    case kDynamicSparseAttentionDuplicateIndex:
      message = "selected_indices contains a duplicate active index";
      break;
    case kDynamicSparseAttentionNonCausalIndex:
      message = "main selected_indices must not refer to a future key";
      break;
    case kDynamicSparseAttentionInvalidSequenceLength:
      message = "seqlens_k values are incompatible with the current sequence and cache capacity";
      break;
    case kDynamicSparseAttentionInvalidPosition:
      message = "a rotary position is outside the rotary cache";
      break;
    default:
      break;
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "DynamicSparseAttention: ", message, ".");
}

template <typename T>
Status LaunchDynamicSparseAttention(
    cudaStream_t stream,
    const DynamicSparseAttentionParameters& parameters,
    const DynamicSparseAttentionData<T>& data,
    bool initialize_key_cache,
    bool initialize_value_cache,
    int max_threads_per_block,
    size_t max_shared_memory_per_block) {
  const int threads = GetThreadsPerBlock(parameters.head_size);
  ORT_RETURN_IF_NOT(threads <= max_threads_per_block,
                    "DynamicSparseAttention: head_size exceeds the CUDA thread-block limit.");

  const int64_t cache_elements =
      static_cast<int64_t>(parameters.batch_size) * parameters.kv_num_heads *
      parameters.cache_capacity * parameters.head_size;
  const size_t cache_bytes =
      SafeInt<size_t>(cache_elements) * sizeof(T);
  ORT_RETURN_IF_NOT(
      data.past_key == nullptr ||
          parameters.past_cache_capacity == parameters.cache_capacity,
      "DynamicSparseAttention: past and present cache capacities must match.");
  if (initialize_key_cache) {
    if (data.past_key == nullptr) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(data.present_key, 0, cache_bytes, stream));
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          data.present_key, data.past_key, cache_bytes, cudaMemcpyDeviceToDevice, stream));
    }
  }
  if (initialize_value_cache) {
    if (data.past_value == nullptr) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(data.present_value, 0, cache_bytes, stream));
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          data.present_value, data.past_value, cache_bytes, cudaMemcpyDeviceToDevice, stream));
    }
  }

  const int64_t query_blocks =
      static_cast<int64_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads;
  const int64_t kv_blocks =
      static_cast<int64_t>(parameters.batch_size) * parameters.sequence_length * parameters.kv_num_heads;
  ORT_RETURN_IF_ERROR(CheckBlockCount(query_blocks, "query preparation"));
  ORT_RETURN_IF_ERROR(CheckBlockCount(kv_blocks, "KV append"));
  const size_t prepare_shared_bytes =
      static_cast<size_t>(threads) * sizeof(float) +
      static_cast<size_t>(parameters.head_size) * sizeof(T);
  const int packed_stride =
      parameters.query_hidden_size + 2 * parameters.kv_hidden_size;

  PrepareQueryKernel<<<static_cast<int>(query_blocks), threads, prepare_shared_bytes, stream>>>(
      data.query, data.prepared_query, data.seqlens_k, data.position_ids,
      data.cos_cache, data.sin_cache, data.q_norm_weight,
      static_cast<int>(query_blocks), parameters.sequence_length, parameters.num_heads,
      parameters.head_size,
      parameters.is_packed_qkv ? packed_stride : parameters.query_hidden_size,
      parameters.rotary_dim, parameters.rotary_max_position, parameters.rotary_offset,
      parameters.rotary_interleaved, parameters.qk_norm_epsilon);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  AppendKvKernel<<<static_cast<int>(kv_blocks), threads, prepare_shared_bytes, stream>>>(
      data.query, data.key, data.value, data.present_key, data.present_value,
      data.seqlens_k, data.position_ids, data.cos_cache, data.sin_cache,
      data.k_norm_weight, static_cast<int>(kv_blocks), parameters.sequence_length,
      parameters.kv_num_heads, parameters.head_size, parameters.query_hidden_size,
      parameters.kv_hidden_size, packed_stride, parameters.cache_capacity,
      parameters.rotary_dim, parameters.rotary_max_position, parameters.rotary_offset,
      parameters.rotary_interleaved, parameters.is_packed_qkv,
      parameters.qk_norm_epsilon);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ORT_RETURN_IF_ERROR(CheckBlockCount(query_blocks, "attention"));
  const bool local_plus_selected =
      parameters.attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected;
  const int64_t candidate_capacity_64 =
      static_cast<int64_t>(parameters.max_selected) +
      (local_plus_selected ? std::min(parameters.local_window_size, parameters.cache_capacity) : 0);
  ORT_RETURN_IF_NOT(candidate_capacity_64 <= std::numeric_limits<int>::max(),
                    "DynamicSparseAttention: candidate count exceeds CUDA kernel limits.");
  const int candidate_capacity = static_cast<int>(candidate_capacity_64);
  const int fused_threads = threads < 128 && max_threads_per_block >= 128 ? 128 : threads;
  const size_t fused_shared_bytes =
      (2 * static_cast<size_t>(candidate_capacity) + static_cast<size_t>(fused_threads)) * sizeof(float) +
      static_cast<size_t>(parameters.head_size) * sizeof(T);
  constexpr int kGroupedDecodeThreads = 32 * kGroupedDecodeQueryHeadsPerBlock;
  constexpr int kGroupedPrefillThreads = 32 * kGroupedPrefillQueryHeadsPerBlock;
  const bool grouped_decode_threads_supported = kGroupedDecodeThreads <= max_threads_per_block;
  const bool grouped_prefill_threads_supported = kGroupedPrefillThreads <= max_threads_per_block;
  if (grouped_decode_threads_supported &&
      UseGroupedDecodeAttention(parameters, sizeof(T), max_shared_memory_per_block)) {
    const int split_count = GetGroupedDecodeSplitCount(parameters);
    const int64_t partial_blocks = query_blocks * split_count;
    const int64_t grouped_blocks =
        static_cast<int64_t>(parameters.batch_size) * parameters.kv_num_heads *
        GetGroupedHeadTiles(parameters, kGroupedDecodeQueryHeadsPerBlock) * split_count;
    ORT_RETURN_IF_ERROR(CheckBlockCount(partial_blocks, "grouped decode partials"));
    ORT_RETURN_IF_ERROR(CheckBlockCount(grouped_blocks, "grouped decode attention"));
    ORT_RETURN_IF_NOT(data.attention_workspace != nullptr,
                      "DynamicSparseAttention: grouped decode workspace is required.");
    float* partial_max = data.attention_workspace;
    float* partial_sum = partial_max + partial_blocks;
    float* partial_output = partial_sum + partial_blocks;
    const size_t grouped_shared_bytes =
        GetGroupedAttentionSharedBytes(parameters, sizeof(T));
    if (parameters.cache_capacity <= 131072) {
      GroupedDecodeDynamicSparseAttentionKernel<T, kGroupedDecodeQueryHeadsPerBlock, true>
          <<<static_cast<int>(grouped_blocks), kGroupedDecodeThreads, grouped_shared_bytes, stream>>>(
              data.prepared_query, data.present_key, data.present_value,
              data.auxiliary_key, data.auxiliary_value, data.selected_indices,
              data.selected_counts, data.seqlens_k, data.head_sink,
              partial_max, partial_sum, partial_output,
              static_cast<int>(grouped_blocks), split_count,
              parameters.num_heads, parameters.kv_num_heads, parameters.head_size,
              parameters.cache_capacity, parameters.auxiliary_sequence_length,
              parameters.max_selected, parameters.local_window_size, parameters.scale,
              local_plus_selected,
              parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
              parameters.use_smooth_softmax);
    } else {
      GroupedDecodeDynamicSparseAttentionKernel<T, kGroupedDecodeQueryHeadsPerBlock, false>
          <<<static_cast<int>(grouped_blocks), kGroupedDecodeThreads, grouped_shared_bytes, stream>>>(
              data.prepared_query, data.present_key, data.present_value,
              data.auxiliary_key, data.auxiliary_value, data.selected_indices,
              data.selected_counts, data.seqlens_k, data.head_sink,
              partial_max, partial_sum, partial_output,
              static_cast<int>(grouped_blocks), split_count,
              parameters.num_heads, parameters.kv_num_heads, parameters.head_size,
              parameters.cache_capacity, parameters.auxiliary_sequence_length,
              parameters.max_selected, parameters.local_window_size, parameters.scale,
              local_plus_selected,
              parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
              parameters.use_smooth_softmax);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    if (split_count <= 32) {
      MergeGroupedDecodeDynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), threads, 0, stream>>>(
          partial_max, partial_sum, partial_output, data.output,
          static_cast<int>(query_blocks), split_count, parameters.head_size);
    } else {
      const size_t merge_shared_bytes = static_cast<size_t>(threads) * sizeof(float);
      MergeDynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), threads, merge_shared_bytes, stream>>>(
          partial_max, partial_sum, partial_output, data.output,
          static_cast<int>(query_blocks), split_count, parameters.head_size);
    }
  } else if (grouped_prefill_threads_supported &&
             UseGroupedPrefillAttention(parameters, sizeof(T), max_shared_memory_per_block)) {
    const int64_t grouped_blocks =
        static_cast<int64_t>(parameters.batch_size) * parameters.sequence_length *
        parameters.kv_num_heads * GetGroupedHeadTiles(parameters, kGroupedPrefillQueryHeadsPerBlock);
    ORT_RETURN_IF_ERROR(CheckBlockCount(grouped_blocks, "grouped prefill attention"));
    const size_t grouped_shared_bytes =
        GetGroupedAttentionSharedBytes(parameters, sizeof(T));
    GroupedPrefillDynamicSparseAttentionKernel<T, kGroupedPrefillQueryHeadsPerBlock>
        <<<static_cast<int>(grouped_blocks), kGroupedPrefillThreads, grouped_shared_bytes, stream>>>(
            data.prepared_query, data.present_key, data.present_value,
            data.auxiliary_key, data.auxiliary_value, data.selected_indices,
            data.selected_counts, data.seqlens_k, data.head_sink, data.output,
            static_cast<int>(grouped_blocks), parameters.sequence_length, parameters.num_heads,
            parameters.kv_num_heads, parameters.head_size, parameters.cache_capacity,
            parameters.auxiliary_sequence_length, parameters.max_selected,
            parameters.local_window_size, parameters.scale, local_plus_selected,
            parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
            parameters.use_smooth_softmax);
  } else if (UseFusedAttention(parameters, sizeof(T), max_shared_memory_per_block)) {
    FusedDynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), fused_threads, fused_shared_bytes, stream>>>(
        data.prepared_query, data.present_key, data.present_value,
        data.auxiliary_key, data.auxiliary_value, data.selected_indices,
        data.selected_counts, data.seqlens_k, data.head_sink, data.output,
        static_cast<int>(query_blocks), parameters.sequence_length, parameters.num_heads,
        parameters.kv_num_heads, parameters.head_size, parameters.cache_capacity,
        parameters.auxiliary_sequence_length, parameters.max_selected,
        parameters.local_window_size, candidate_capacity, parameters.scale,
        local_plus_selected,
        parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
        parameters.use_smooth_softmax);
  } else {
    const int split_count = GetAttentionSplitCount(parameters);
    const int64_t partial_blocks = query_blocks * split_count;
    ORT_RETURN_IF_ERROR(CheckBlockCount(partial_blocks, "split attention"));
    ORT_RETURN_IF_NOT(split_count == 1 || data.attention_workspace != nullptr,
                      "DynamicSparseAttention: split attention workspace is required.");
    float* partial_max = data.attention_workspace;
    float* partial_sum = split_count == 1 ? nullptr : partial_max + partial_blocks;
    float* partial_output = split_count == 1 ? nullptr : partial_sum + partial_blocks;
    const int split_threads = threads < 128 && max_threads_per_block >= 128 ? 128 : threads;
    const size_t split_shared_bytes =
        (2 * static_cast<size_t>(kSplitCandidateSize) + static_cast<size_t>(split_threads)) * sizeof(float) +
        static_cast<size_t>(parameters.head_size) * sizeof(T);
    ORT_RETURN_IF_NOT(split_shared_bytes < max_shared_memory_per_block,
                      "DynamicSparseAttention: split attention exceeds the CUDA shared-memory limit.");
    SplitDynamicSparseAttentionKernel<<<static_cast<int>(partial_blocks), split_threads, split_shared_bytes, stream>>>(
        data.prepared_query, data.present_key, data.present_value,
        data.auxiliary_key, data.auxiliary_value, data.selected_indices,
        data.selected_counts, data.seqlens_k, data.head_sink,
        partial_max, partial_sum, partial_output, data.output,
        static_cast<int>(partial_blocks), split_count,
        parameters.sequence_length, parameters.num_heads,
        parameters.kv_num_heads, parameters.head_size, parameters.cache_capacity,
        parameters.auxiliary_sequence_length, parameters.max_selected,
        parameters.local_window_size, parameters.scale,
        local_plus_selected,
        parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
        parameters.use_smooth_softmax);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    if (split_count > 1) {
      const size_t merge_shared_bytes = static_cast<size_t>(threads) * sizeof(float);
      MergeDynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), threads, merge_shared_bytes, stream>>>(
          partial_max, partial_sum, partial_output, data.output,
          static_cast<int>(query_blocks), split_count, parameters.head_size);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchDynamicSparseAttention<float>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<float>&, bool, bool, int, size_t);
template Status LaunchDynamicSparseAttention<half>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<half>&, bool, bool, int, size_t);
template Status LaunchDynamicSparseAttention<__nv_bfloat16>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<__nv_bfloat16>&, bool, bool, int, size_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
