// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void LaunchInitKernel(
    float* beam_scores,
    int batch_size,
    int num_beams,
    cudaStream_t stream);

template <typename T>
void LaunchAddProbsKernel(T* log_probs,
                          T* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream);

template <typename T>
void LaunchLogitsProcessKernel(
    T* next_token_scores,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    int* presence_mask,
    float presence_penalty,
    float temperature,
    int batch_size,
    int num_beams,
    int vocab_size,
    int padded_vocab_size,
    int demote_token_id,
    int32_t* sequences,
    int max_sequence_length,
    int current_sequence_length,
    float repetition_penalty,
    int no_repeat_ngram_size,
    cudaStream_t stream);

void LaunchNextTokenKernel(const int64_t* next_token_indices,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           int batch_size,
                           int top_k,
                           int vocab_size,
                           cudaStream_t stream);

void LaunchUpdateGptKernel(const int32_t* old_mask_data,
                           int32_t* mask_data,
                           int32_t* next_positions,
                           int batch_beam_size,
                           int current_length,
                           cudaStream_t stream);

template <typename T>
void GetTempStorageSize(const T *d_keys_in,
                        const int* d_values_in,
                        int* d_offsets,
                        int num_items,
                        int num_segments,
                        cudaStream_t stream,
                        bool is_descending,
                        size_t& temp_storage_bytes);

void LaunchSetupParamsKernel(int* d_values_in,
                             int* d_offsets,
                             int batch_size,
                             int vocab_size,
                             cudaStream_t stream);

template <typename T>
void LaunchSortPairs(void *d_temp_storage,
                     size_t temp_storage_bytes,
                     const T *d_keys_in,
                     T *d_keys_out,
                     const int *d_values_in,
                     int *d_values_out,
                     int num_items,
                     int num_segments,
                     int *d_offsets,
                     cudaStream_t stream,
                     bool is_descending);

template <typename T>
void LaunchFilterLogitsKernel(float* d_sorted_logits_in,
                              const int* d_sorted_indices,
                              T* d_logits_in_out,
                              float top_p,
                              float filter_value,
                              int min_tokens_to_keep,
                              int batch_size,
                              int vocab_size,
                              cudaStream_t stream,
                              bool is_descending);

void TorchMultinomialKernelLauncher(float* d_input,
                                    float* d_sampled,
                                    int32_t* d_output,
                                    int batch_size,
                                    int vocab_size,
                                    int* d_presence_mask,
                                    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
