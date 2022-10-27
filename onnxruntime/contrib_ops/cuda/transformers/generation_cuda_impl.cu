// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cub/util_type.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>
#include "contrib_ops/cuda/transformers/generation_cuda_impl.h"


namespace onnxruntime {
namespace contrib {
namespace cuda {
__global__ void InitKernel(float* beam_scores,
                           int num_beams,
                           int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    int beam_index = index % num_beams;
    beam_scores[index] = beam_index > 0 ? static_cast<float>(-1e9) : 0.0f;
  }
}

void LaunchInitKernel(
    float* beam_scores,
    int batch_size,
    int num_beams,
    cudaStream_t stream) {
  int total_elements = batch_size * num_beams;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  InitKernel<<<gridSize, blockSize, 0, stream>>>(beam_scores, num_beams, total_elements);
}

__global__ void NextTokenKernel(const int64_t* next_token_indices,
                                int32_t* next_indices,
                                int32_t* next_tokens,
                                int vocab_size,
                                int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    next_indices[index] = next_token_indices[index] / vocab_size;
    next_tokens[index] = next_token_indices[index] % vocab_size;
  }
}

void LaunchNextTokenKernel(const int64_t* next_token_indices,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           int batch_size,
                           int top_k,
                           int vocab_size,
                           cudaStream_t stream) {
  int total_elements = batch_size * top_k;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  NextTokenKernel<<<gridSize, blockSize, 0, stream>>>(next_token_indices,
                                                      next_indices,
                                                      next_tokens,
                                                      vocab_size,
                                                      total_elements);
}

template <typename T>
__global__ void LogitsProcessKernel(
    T* next_token_scores,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    const int* presence_mask,
    float presence_penalty,
    float temperature,
    int num_beams,
    int vocab_size,
    int total_elements,
    int demote_token_id,
    int32_t* sequences,
    int max_sequence_length,
    int current_sequence_length,
    float repetition_penalty,
    int no_repeat_ngram_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    int batch_beam_index = index / vocab_size;
    int word_id = index % vocab_size;

    // RepetitionPenaltyLogitsProcessor
    if (repetition_penalty != 1.0f) {
      int32_t* current_sequence = sequences + batch_beam_index * max_sequence_length;
      bool found = false;
      for (int i = 0; i < current_sequence_length; i++) {
        if (current_sequence[i] == word_id) {
          found = true;
          break;
        }
      }
      if (found) {
        float score = (float)next_token_scores[index];
        next_token_scores[index] = (T)(score < 0 ? score * repetition_penalty : score / repetition_penalty);
      }
    }

    // NoRepeatNGramLogitsProcessor
    if (no_repeat_ngram_size > 0 && current_sequence_length >= no_repeat_ngram_size) {
      int32_t* current_sequence = sequences + batch_beam_index * max_sequence_length;
      bool found = false;
      for (int i = no_repeat_ngram_size - 1; i < current_sequence_length; i++) {
        if (current_sequence[i] == word_id) {  // last token of n-gram matched
          found = true;
          for (int j = 0; j < no_repeat_ngram_size - 1; j++) {  // match the remaining N-1 tokens
            if (current_sequence[i - j - 1] != current_sequence[current_sequence_length - 1 - j]) {
              found = false;
              break;
            }
          }
          if (found) {
            break;
          }
        }
      }

      if (found) {
        next_token_scores[index] = cub::FpLimits<T>::Lowest();
        return;
      }
    }

    // VocabMaskLogitsProcessor
    if (vocab_mask != nullptr && vocab_mask[word_id] == 0) {
      next_token_scores[index] = cub::FpLimits<T>::Lowest();
      return;
    }

    // PrefixVocabMaskLogitsProcessor
    int batch_id = batch_beam_index / num_beams;
    if (prefix_vocab_mask != nullptr && prefix_vocab_mask[batch_id * vocab_size + word_id] == 0) {
      next_token_scores[index] = cub::FpLimits<T>::Lowest();
      return;
    }

    // MinLengthLogitsProcessor
    if (word_id == demote_token_id) {
      next_token_scores[index] = cub::FpLimits<T>::Lowest();
    }

    // PresencePenaltyLogitsProcessor
    if (presence_mask != nullptr && presence_mask[index] == 1) {
      float score = (float)next_token_scores[index] - presence_penalty;
      next_token_scores[index] = (T)score;
    }

    // TemperatureLogitsProcessor
    if (temperature != 1.0f) {
      float score = (float)(next_token_scores[index]);
      next_token_scores[index] = (T)(score / temperature);
    }
  }
}

template <typename T>
void LaunchLogitsProcessKernel(
    T* next_token_scores,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    const int* presence_mask,
    float presence_penalty,
    float temperature,
    int batch_size,
    int num_beams,
    int vocab_size,
    int demote_token_id,
    int32_t* sequences,
    int max_sequence_length,
    int current_sequence_length,
    float repetition_penalty,
    int no_repeat_ngram_size,
    cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  LogitsProcessKernel<T><<<gridSize, blockSize, 0, stream>>>(
      next_token_scores,
      vocab_mask,
      prefix_vocab_mask,
      presence_mask,
      presence_penalty,
      temperature,
      num_beams,
      vocab_size,
      total_elements,
      demote_token_id,
      sequences,
      max_sequence_length,
      current_sequence_length,
      repetition_penalty,
      no_repeat_ngram_size);
}

// Instantiation
template void LaunchLogitsProcessKernel(
    float* next_token_scores,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    const int* presence_mask,
    float presence_penalty,
    float temperature,
    int batch_size,
    int num_beams,
    int vocab_size,
    int demote_token_id,
    int32_t* sequences,
    int max_sequence_length,
    int current_sequence_length,
    float repetition_penalty,
    int no_repeat_ngram_size,
    cudaStream_t stream);

template void LaunchLogitsProcessKernel(
    half* next_token_scores,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    const int* presence_mask,
    float presence_penalty,
    float temperature,
    int batch_size,
    int num_beams,
    int vocab_size,
    int demote_token_id,
    int32_t* sequences,
    int max_sequence_length,
    int current_sequence_length,
    float repetition_penalty,
    int no_repeat_ngram_size,
    cudaStream_t stream);

__global__ void AddProbsKernel(float* log_probs,
                               float* cum_log_probs,
                               const int vocab_size,
                               const int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_beam_index = index / vocab_size;

  if (index < total_elements)
    log_probs[index] += cum_log_probs[batch_beam_index];
}

template <typename T>
void LaunchAddProbsKernel(T* log_probs,
                          T* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  AddProbsKernel<<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}

template void LaunchAddProbsKernel(
    float* log_probs,
    float* cum_log_probs,
    const int batch_size,
    const int num_beams,
    const int vocab_size,
    cudaStream_t stream);

template <typename T>
__global__ void UpdateGptInputsKernel(const T* old_mask_data,
                                      T* mask_data,
                                      int32_t* next_positions,
                                      int batch_beam_size,
                                      int current_length) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < batch_beam_size * current_length) {
    // Update attention mask.
    int i = index / current_length;
    int j = index % current_length;
    mask_data[index] = (j < current_length - 1) ? old_mask_data[i * (current_length - 1) + j] : static_cast<T>(1);

    if (next_positions != nullptr) {
      // Update sequence length (or next positions).
      if (index < batch_beam_size) {
        next_positions[index]++;
      }
    }
  }
}

void LaunchUpdateGptKernel(const int32_t* old_mask_data,
                           int32_t* mask_data,
                           int32_t* next_positions,
                           int batch_beam_size,
                           int current_length,
                           cudaStream_t stream) {
  assert(current_length > 0);
  int total_elements = batch_beam_size * current_length;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  UpdateGptInputsKernel<int32_t><<<gridSize, blockSize, 0, stream>>>(
      old_mask_data, mask_data, next_positions, batch_beam_size, current_length);
}

// bugbug: merge those kernels into one
template <typename T>
size_t GetTempStorageSize(const T *d_keys_in,
                          const int* d_values_in,
                          int* d_offsets,
                          int num_items,
                          int num_segments,
                          cudaStream_t stream) {
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       (T*)nullptr,
                                                       d_values_in,
                                                       (int*)nullptr,
                                                       num_items,
                                                       num_segments,
                                                       d_offsets,
                                                       d_offsets + 1,
                                                       0,
                                                       sizeof(T) * 8,
                                                       stream);
    return temp_storage_bytes;
}

template size_t GetTempStorageSize(
  const float *d_keys_in,
  const int* d_values_in,
  int* d_offsets,
  int num_items,
  int num_segments,
  cudaStream_t stream);

template size_t GetTempStorageSize(
  const half *d_keys_in,
  const int* d_values_in,
  int* d_offsets,
  int num_items,
  int num_segments,
  cudaStream_t stream);

// bugbug: merge to one kernel
__global__ void SetupParamsKernel(int* d_values_in,
                                  int* d_offsets,
                                  int batch_size,
                                  int vocab_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * vocab_size;
  if (index < total_elements) {
    d_values_in[index] = index % vocab_size;
  }
  if (index < batch_size + 1) {
    d_offsets[index] = index * vocab_size;
  }
}

void LaunchSetupParamsKernel(int* d_values_in,
                             int* d_offsets,
                             int batch_size,
                             int vocab_size,
                             cudaStream_t stream) {
  int total_elements = batch_size * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  SetupParamsKernel<<<gridSize, blockSize, 0, stream>>>(d_values_in,
                                                        d_offsets,
                                                        batch_size,
                                                        vocab_size);
}

template <typename T>
void LaunchSortPairsDescending(void *d_temp_storage,
                               size_t temp_storage_bytes,
                               const T *d_keys_in,
                               T *d_keys_out,
                               const int *d_values_in,
                               int *d_values_out,
                               int num_items,
                               int num_segments,
                               int *d_offsets,
                               cudaStream_t stream) {
  cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_keys_in,
                                                     d_keys_out,
                                                     d_values_in,
                                                     d_values_out,
                                                     num_items,
                                                     num_segments,
                                                     d_offsets,
                                                     d_offsets + 1,
                                                     0,
                                                     sizeof(T) * 8,
                                                     stream);
}

template void LaunchSortPairsDescending(void *d_temp_storage,
                                        size_t temp_storage_bytes,
                                        const float *d_keys_in,
                                        float *d_keys_out,
                                        const int *d_values_in,
                                        int *d_values_out,
                                        int num_items,
                                        int num_segments,
                                        int *d_offsets,
                                        cudaStream_t stream);

template void LaunchSortPairsDescending(void *d_temp_storage,
                                        size_t temp_storage_bytes,
                                        const half *d_keys_in,
                                        half *d_keys_out,
                                        const int *d_values_in,
                                        int *d_values_out,
                                        int num_items,
                                        int num_segments,
                                        int *d_offsets,
                                        cudaStream_t stream);

// A trick here: cumuliative sum of the sorted logits is a temporarily variable in the kernel.
template <typename T>
__global__ void FilterLogitsKernel(float* d_sorted_logits_in,
                                   const int* d_sorted_indices,
                                   T* d_logits_in_out,
                                   float top_p,
                                   float filter_value,
                                   int batch_size,
                                   int vocab_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int vocab_idx = index % vocab_size;
  int batch_id = index / vocab_size;
  int start_index = batch_id * vocab_size;

  int count = vocab_idx;
  float sum = 0.0f;
  while (count != 0) {
    sum += d_sorted_logits_in[start_index];
    ++start_index;
    --count;
  }

  if (sum > top_p) {
    // Shift the indices to the right by one according to the Turing implementation.
    int shifted_index = index + 1;
    if (shifted_index % vocab_size != 0) {
      int original_index = batch_id * vocab_size + d_sorted_indices[shifted_index];
      d_logits_in_out[original_index] = (T)filter_value;
    }
  }
}

template <typename T>
void LaunchFilterLogitsKernel(float* d_sorted_logits_in,
                              const int* d_sorted_indices,
                              T* d_logits_in_out,
                              float top_p,
                              float filter_value,
                              int batch_size,
                              int vocab_size,
                              cudaStream_t stream) {
  int total_elements = batch_size * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  FilterLogitsKernel<<<gridSize, blockSize, 0, stream>>>(d_sorted_logits_in,
                                                         d_sorted_indices,
                                                         d_logits_in_out,
                                                         top_p,
                                                         filter_value,
                                                         batch_size,
                                                         vocab_size);
}

template void LaunchFilterLogitsKernel(float* d_sorted_logits_in,
                                       const int* d_sorted_indices,
                                       float* d_logits_in_out,
                                       float top_p,
                                       float filter_value,
                                       int batch_size,
                                       int vocab_size,
                                       cudaStream_t stream);

template void LaunchFilterLogitsKernel(float* d_sorted_logits_in,
                                       const int* d_sorted_indices,
                                       half* d_logits_in_out,
                                       float top_p,
                                       float filter_value,
                                       int batch_size,
                                       int vocab_size,
                                       cudaStream_t stream);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
