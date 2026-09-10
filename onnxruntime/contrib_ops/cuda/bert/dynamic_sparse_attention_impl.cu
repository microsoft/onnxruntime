// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

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
                                     int32_t* error_flag) {
  const int row = static_cast<int>(blockIdx.x);
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  if (b >= batch_size || threadIdx.x != 0) {
    return;
  }

  const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
  if (total_length_64 < sequence_length ||
      total_length_64 > maximum_total_length ||
      total_length_64 > main_capacity) {
    SetValidationError(error_flag, kDynamicSparseAttentionInvalidSequenceLength);
    return;
  }
  const int total_length = static_cast<int>(total_length_64);

  if (do_rotary) {
    const int64_t position = position_ids == nullptr
                                 ? static_cast<int64_t>(total_length - sequence_length + s)
                                 : position_ids[row];
    if (position < 0 || position >= rotary_max_position) {
      SetValidationError(error_flag, kDynamicSparseAttentionInvalidPosition);
      return;
    }
  }

  const int count = selected_counts[row];
  if (count < 0 || count > max_selected) {
    SetValidationError(error_flag, kDynamicSparseAttentionInvalidCount);
    return;
  }

  const int source_length = use_auxiliary ? auxiliary_sequence_length : total_length;
  const int query_position = total_length - sequence_length + s;
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;
  for (int i = 0; i < max_selected; ++i) {
    const int index = row_indices[i];
    if (i >= count) {
      if (index != -1) {
        SetValidationError(error_flag, kDynamicSparseAttentionInvalidPadding);
        return;
      }
      continue;
    }
    if (index < 0) {
      SetValidationError(error_flag, kDynamicSparseAttentionNegativeIndex);
      return;
    }
    if (index >= source_length) {
      SetValidationError(error_flag, kDynamicSparseAttentionIndexOutOfBounds);
      return;
    }
    if (!use_auxiliary && index > query_position) {
      SetValidationError(error_flag, kDynamicSparseAttentionNonCausalIndex);
      return;
    }
    for (int j = 0; j < i; ++j) {
      if (row_indices[j] == index) {
        SetValidationError(error_flag, kDynamicSparseAttentionDuplicateIndex);
        return;
      }
    }
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
__device__ __forceinline__ void AccumulateCandidate(const T* query,
                                                    const T* key,
                                                    const T* value,
                                                    int head_size,
                                                    float scale,
                                                    float* reduction,
                                                    float& accumulator,
                                                    float& max_logit,
                                                    float& denominator,
                                                    float& old_weight,
                                                    float& new_weight) {
  const int h = static_cast<int>(threadIdx.x);
  float partial = 0.0f;
  if (h < head_size) {
    partial = ToFloat(query[h]) * ToFloat(key[h]);
  }
  reduction[h] = partial;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (h < static_cast<int>(stride)) {
      reduction[h] += reduction[h + stride];
    }
    __syncthreads();
  }

  if (h == 0) {
    const float logit = reduction[0] * scale;
    const float next_max = fmaxf(max_logit, logit);
    old_weight = denominator == 0.0f ? 0.0f : expf(max_logit - next_max);
    new_weight = expf(logit - next_max);
    denominator = denominator * old_weight + new_weight;
    max_logit = next_max;
  }
  __syncthreads();
  if (h < head_size) {
    accumulator = accumulator * old_weight + new_weight * ToFloat(value[h]);
  }
  __syncthreads();
}

template <typename T>
__global__ void DynamicSparseAttentionKernel(const T* query,
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
  const int head = block % num_heads;
  const int row = block / num_heads;
  const int b = row / sequence_length;
  const int s = row - b * sequence_length;
  const int h = static_cast<int>(threadIdx.x);
  const int kv_head = head / (num_heads / kv_num_heads);
  const int64_t total_length_64 = static_cast<int64_t>(seqlens_k[b]) + 1;
  const bool valid_length = total_length_64 >= sequence_length && total_length_64 <= main_capacity;
  const int total_length = valid_length ? static_cast<int>(total_length_64) : 0;
  const int query_position = valid_length ? total_length - sequence_length + s : -1;
  const T* query_head = query + (static_cast<int64_t>(row) * num_heads + head) * head_size;

  extern __shared__ float reduction[];
  __shared__ float max_logit;
  __shared__ float denominator;
  __shared__ float old_weight;
  __shared__ float new_weight;
  if (h == 0) {
    if (use_smooth_softmax) {
      max_logit = head_sink == nullptr ? 0.0f : ToFloat(head_sink[head]);
      denominator = 1.0f;
    } else {
      max_logit = -CUDART_INF_F;
      denominator = 0.0f;
    }
    old_weight = 0.0f;
    new_weight = 0.0f;
  }
  __syncthreads();

  float accumulator = 0.0f;
  int local_start = 0;
  if (local_plus_selected && query_position >= 0) {
    local_start = query_position - local_window_size + 1;
    local_start = local_start < 0 ? 0 : local_start;
    const int available_length = total_length < main_capacity ? total_length : main_capacity;
    const int local_end = query_position < available_length - 1 ? query_position : available_length - 1;
    for (int index = local_start; index <= local_end; ++index) {
      const int64_t cache_offset =
          ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
      AccumulateCandidate(query_head, main_key + cache_offset, main_value + cache_offset,
                          head_size, scale, reduction, accumulator,
                          max_logit, denominator, old_weight, new_weight);
    }
  }

  int count = selected_counts[row];
  count = count < 0 ? 0 : (count > max_selected ? max_selected : count);
  const int32_t* row_indices =
      max_selected == 0 ? selected_indices : selected_indices + static_cast<int64_t>(row) * max_selected;
  for (int i = 0; i < count; ++i) {
    const int index = row_indices[i];
    bool valid = index >= 0;
    if (selected_from_auxiliary) {
      valid = valid && index < auxiliary_sequence_length;
    } else {
      valid = valid && index < total_length && index <= query_position && index < main_capacity;
      if (local_plus_selected && index >= local_start && index <= query_position) {
        valid = false;
      }
    }
    if (!valid) {
      continue;
    }

    const T* key_head;
    const T* value_head;
    if (selected_from_auxiliary) {
      const int64_t offset =
          ((static_cast<int64_t>(b) * kv_num_heads + kv_head) *
               auxiliary_sequence_length +
           index) *
          head_size;
      key_head = auxiliary_key + offset;
      value_head = auxiliary_value + offset;
    } else {
      const int64_t offset =
          ((static_cast<int64_t>(b) * kv_num_heads + kv_head) * main_capacity + index) * head_size;
      key_head = main_key + offset;
      value_head = main_value + offset;
    }
    AccumulateCandidate(query_head, key_head, value_head, head_size, scale, reduction,
                        accumulator, max_logit, denominator, old_weight, new_weight);
  }

  if (h < head_size) {
    const int64_t output_offset =
        (static_cast<int64_t>(row) * num_heads + head) * head_size + h;
    output[output_offset] =
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

}  // namespace

Status ValidateDynamicSparseAttentionOnDevice(
    cudaStream_t stream,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    const DynamicSparseAttentionParameters& parameters,
    int32_t* error_flag,
    bool copy_result_to_host) {
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(error_flag, 0, sizeof(int32_t), stream));
  const int64_t row_count =
      static_cast<int64_t>(parameters.batch_size) * parameters.sequence_length;
  ORT_RETURN_IF_ERROR(CheckBlockCount(row_count, "validation"));
  ValidateInputsKernel<<<static_cast<int>(row_count), 1, 0, stream>>>(
      selected_indices, selected_counts, seqlens_k, position_ids,
      parameters.batch_size, parameters.sequence_length, parameters.max_selected,
      parameters.total_sequence_length,
      parameters.cache_capacity, parameters.auxiliary_sequence_length,
      parameters.rotary_max_position, parameters.do_rotary,
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
      error_flag);
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
    int max_threads_per_block) {
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
  const size_t attention_shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  DynamicSparseAttentionKernel<<<static_cast<int>(query_blocks), threads, attention_shared_bytes, stream>>>(
      data.prepared_query, data.present_key, data.present_value,
      data.auxiliary_key, data.auxiliary_value, data.selected_indices,
      data.selected_counts, data.seqlens_k, data.head_sink, data.output,
      static_cast<int>(query_blocks), parameters.sequence_length, parameters.num_heads,
      parameters.kv_num_heads, parameters.head_size, parameters.cache_capacity,
      parameters.auxiliary_sequence_length, parameters.max_selected,
      parameters.local_window_size, parameters.scale,
      parameters.attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected,
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary,
      parameters.use_smooth_softmax);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchDynamicSparseAttention<float>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<float>&, bool, bool, int);
template Status LaunchDynamicSparseAttention<half>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<half>&, bool, bool, int);
template Status LaunchDynamicSparseAttention<__nv_bfloat16>(
    cudaStream_t, const DynamicSparseAttentionParameters&,
    const DynamicSparseAttentionData<__nv_bfloat16>&, bool, bool, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
