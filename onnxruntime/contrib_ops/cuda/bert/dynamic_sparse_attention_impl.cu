// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <limits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kSplitCandidateSize = 256;
constexpr size_t kMaxFusedSharedBytes = 16 * 1024;
constexpr size_t kMaxValidationSharedBytes = 32 * 1024;

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
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
                                     uint32_t* validation_bitmap,
                                     size_t validation_bitmap_words,
                                     bool use_shared_bitmap,
                                     int32_t* error_flag) {
  const int row = static_cast<int>(blockIdx.x);
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  if (b >= batch_size) {
    return;
  }

  extern __shared__ uint32_t shared_bitmap[];
  if (use_shared_bitmap) {
    for (size_t i = threadIdx.x; i < validation_bitmap_words; i += blockDim.x) {
      shared_bitmap[i] = 0;
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
  uint32_t* row_bitmap =
      use_shared_bitmap
          ? shared_bitmap
          : validation_bitmap + static_cast<size_t>(row) * validation_bitmap_words;
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
      const uint32_t mask = 1U << (index & 31);
      const uint32_t previous = atomicOr(row_bitmap + (index >> 5), mask);
      if ((previous & mask) != 0) {
        error = kDynamicSparseAttentionDuplicateIndex;
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
__global__ void InitializeCacheKernel(T* present_key,
                                      T* present_value,
                                      const T* past_key,
                                      const T* past_value,
                                      int cache_capacity,
                                      int past_capacity,
                                      int head_size,
                                      bool initialize_key,
                                      bool initialize_value,
                                      int64_t element_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }

  const int64_t row_width = static_cast<int64_t>(cache_capacity) * head_size;
  const int64_t row = index / row_width;
  const int64_t within_row = index - row * row_width;
  const int sequence = static_cast<int>(within_row / head_size);
  const int head_offset = static_cast<int>(within_row - static_cast<int64_t>(sequence) * head_size);
  const int64_t source_index =
      row * static_cast<int64_t>(past_capacity) * head_size +
      static_cast<int64_t>(sequence) * head_size + head_offset;

  if (initialize_key) {
    present_key[index] = past_key != nullptr && sequence < past_capacity
                             ? past_key[source_index]
                             : FromFloat<T>(0.0f);
  }
  if (initialize_value) {
    present_value[index] = past_value != nullptr && sequence < past_capacity
                               ? past_value[source_index]
                               : FromFloat<T>(0.0f);
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
  float* reduction = logits + candidate_capacity;
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
      logits[candidate] = valid ? dot * scale : -FLT_MAX;
    }
  }
  __syncthreads();

  float thread_max = use_smooth_softmax && h == 0
                         ? (head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]))
                         : -FLT_MAX;
  for (int candidate = h; candidate < candidate_count; candidate += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, logits[candidate]);
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
    const float weight = logit == -FLT_MAX ? 0.0f : expf(logit - max_logit);
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
      if (weight == 0.0f) {
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
  const int candidate_start = split * kSplitCandidateSize;
  const int split_candidate_count = max(0, min(kSplitCandidateSize, candidate_count - candidate_start));
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;

  extern __shared__ float shared[];
  float* logits = shared;
  float* reduction = logits + kSplitCandidateSize;
  T* shared_query = reinterpret_cast<T*>(reduction + blockDim.x);
  if (h < head_size) {
    shared_query[h] = query_head[h];
  }
  __syncthreads();

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
      logits[split_candidate] = valid ? dot * scale : -FLT_MAX;
    }
  }
  __syncthreads();

  float thread_max = use_smooth_softmax && split == 0 && h == 0
                         ? (head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]))
                         : -FLT_MAX;
  for (int candidate = h; candidate < split_candidate_count;
       candidate += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, logits[candidate]);
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
  if (use_smooth_softmax && split == 0 && h == 0) {
    const float sink = head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]);
    thread_sum = expf(sink - max_logit);
  }
  for (int candidate = h; candidate < split_candidate_count;
       candidate += static_cast<int>(blockDim.x)) {
    const float logit = logits[candidate];
    const float weight = logit == -FLT_MAX ? 0.0f : expf(logit - max_logit);
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
  if (h == 0) {
    partial_max[partial_block] = max_logit;
    partial_sum[partial_block] = reduction[0];
  }
  __syncthreads();

  if (h < head_size) {
    float accumulator = 0.0f;
    for (int split_candidate = 0; split_candidate < split_candidate_count; ++split_candidate) {
      const float weight = logits[split_candidate];
      if (weight == 0.0f) {
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
      accumulator += weight * ToFloat(value_head[h]);
    }
    partial_output[static_cast<int64_t>(partial_block) * head_size + h] = accumulator;
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
    if (split_max != -FLT_MAX) {
      thread_sum += partial_sum[partial_start + split] * expf(split_max - max_logit);
    }
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
      if (split_max != -FLT_MAX) {
        const float merge_scale = expf(split_max - max_logit);
        const int64_t partial_offset =
            (static_cast<int64_t>(partial_start + split) * head_size) + h;
        accumulator += partial_output[partial_offset] * merge_scale;
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

bool UseFusedAttention(const DynamicSparseAttentionParameters& parameters,
                       size_t element_size,
                       size_t max_shared_memory_per_block) {
  if (GetCandidateCapacity(parameters) > kSplitCandidateSize) {
    return false;
  }

  const int threads = GetThreadsPerBlock(parameters.head_size);
  const int fused_threads = threads < 128 ? 128 : threads;
  const size_t fused_shared_bytes =
      (static_cast<size_t>(GetCandidateCapacity(parameters)) + static_cast<size_t>(fused_threads)) * sizeof(float) +
      static_cast<size_t>(parameters.head_size) * element_size;
  return fused_shared_bytes <= kMaxFusedSharedBytes &&
         fused_shared_bytes < max_shared_memory_per_block;
}

size_t GetValidationBitmapWords(const DynamicSparseAttentionParameters& parameters) {
  if (parameters.max_selected == 0) {
    return 0;
  }

  const size_t source_length =
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary
          ? static_cast<size_t>(parameters.auxiliary_sequence_length)
          : static_cast<size_t>(parameters.cache_capacity);
  return (source_length + 31) / 32;
}

bool UseSharedValidationBitmap(const DynamicSparseAttentionParameters& parameters,
                               size_t max_shared_memory_per_block) {
  const size_t shared_bytes = GetValidationBitmapWords(parameters) * sizeof(uint32_t);
  return shared_bytes <= kMaxValidationSharedBytes &&
         shared_bytes < max_shared_memory_per_block;
}

}  // namespace

size_t GetDynamicSparseAttentionValidationWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t max_shared_memory_per_block) {
  if (UseSharedValidationBitmap(parameters, max_shared_memory_per_block)) {
    return 0;
  }

  const size_t row_count =
      SafeInt<size_t>(parameters.batch_size) * parameters.sequence_length;
  return SafeInt<size_t>(row_count) * GetValidationBitmapWords(parameters);
}

size_t GetDynamicSparseAttentionWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t element_size,
    size_t max_shared_memory_per_block) {
  if (UseFusedAttention(parameters, element_size, max_shared_memory_per_block)) {
    return 0;
  }

  const size_t query_blocks =
      static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads;
  const size_t split_count =
      (static_cast<size_t>(GetCandidateCapacity(parameters)) + kSplitCandidateSize - 1) /
      kSplitCandidateSize;
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
  const size_t validation_bitmap_words = GetValidationBitmapWords(parameters);
  const bool use_shared_bitmap =
      UseSharedValidationBitmap(parameters, max_shared_memory_per_block);
  const size_t validation_bitmap_elements =
      use_shared_bitmap ? 0 : static_cast<size_t>(row_count) * validation_bitmap_words;
  if (validation_bitmap_elements > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        validation_bitmap, 0, validation_bitmap_elements * sizeof(uint32_t), stream));
  }
  ORT_RETURN_IF_ERROR(CheckBlockCount(row_count, "validation"));
  constexpr int kValidationThreads = 256;
  const size_t validation_shared_bytes =
      use_shared_bitmap ? validation_bitmap_words * sizeof(uint32_t) : 0;
  ValidateInputsKernel<<<static_cast<int>(row_count), kValidationThreads,
                         validation_shared_bytes, stream>>>(
      selected_indices, selected_counts, seqlens_k, position_ids,
      parameters.batch_size, parameters.sequence_length, parameters.max_selected,
      parameters.total_sequence_length,
      parameters.cache_capacity, parameters.auxiliary_sequence_length,
      parameters.rotary_max_position, parameters.do_rotary,
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
      validation_bitmap, validation_bitmap_words, use_shared_bitmap, error_flag);
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
  if (initialize_key_cache || initialize_value_cache) {
    constexpr int kCopyThreads = 256;
    const int64_t copy_blocks = (cache_elements + kCopyThreads - 1) / kCopyThreads;
    ORT_RETURN_IF_ERROR(CheckBlockCount(copy_blocks, "cache initialization"));
    InitializeCacheKernel<<<static_cast<int>(copy_blocks), kCopyThreads, 0, stream>>>(
        data.present_key, data.present_value, data.past_key, data.past_value,
        parameters.cache_capacity,
        parameters.past_cache_capacity, parameters.head_size,
        initialize_key_cache, initialize_value_cache, cache_elements);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
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
      (static_cast<size_t>(candidate_capacity) + static_cast<size_t>(fused_threads)) * sizeof(float) +
      static_cast<size_t>(parameters.head_size) * sizeof(T);
  if (UseFusedAttention(parameters, sizeof(T), max_shared_memory_per_block)) {
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
    const int split_count =
        (candidate_capacity + kSplitCandidateSize - 1) / kSplitCandidateSize;
    const int64_t partial_blocks = query_blocks * split_count;
    ORT_RETURN_IF_ERROR(CheckBlockCount(partial_blocks, "split attention"));
    ORT_RETURN_IF_NOT(data.attention_workspace != nullptr,
                      "DynamicSparseAttention: split attention workspace is required.");
    float* partial_max = data.attention_workspace;
    float* partial_sum = partial_max + partial_blocks;
    float* partial_output = partial_sum + partial_blocks;
    const int split_threads = threads < 128 && max_threads_per_block >= 128 ? 128 : threads;
    const size_t split_shared_bytes =
        (static_cast<size_t>(kSplitCandidateSize) + static_cast<size_t>(split_threads)) * sizeof(float) +
        static_cast<size_t>(parameters.head_size) * sizeof(T);
    ORT_RETURN_IF_NOT(split_shared_bytes < max_shared_memory_per_block,
                      "DynamicSparseAttention: split attention exceeds the CUDA shared-memory limit.");
    SplitDynamicSparseAttentionKernel<<<static_cast<int>(partial_blocks), split_threads, split_shared_bytes, stream>>>(
        data.prepared_query, data.present_key, data.present_value,
        data.auxiliary_key, data.auxiliary_value, data.selected_indices,
        data.selected_counts, data.seqlens_k, data.head_sink,
        partial_max, partial_sum, partial_output,
        static_cast<int>(partial_blocks), split_count,
        parameters.sequence_length, parameters.num_heads,
        parameters.kv_num_heads, parameters.head_size, parameters.cache_capacity,
        parameters.auxiliary_sequence_length, parameters.max_selected,
        parameters.local_window_size, parameters.scale,
        local_plus_selected,
        parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
        parameters.use_smooth_softmax);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    const size_t merge_shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    MergeDynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), threads, merge_shared_bytes, stream>>>(
        partial_max, partial_sum, partial_output, data.output,
        static_cast<int>(query_blocks), split_count, parameters.head_size);
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
