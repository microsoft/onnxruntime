// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchAddProbsKernel(T* log_probs,
                          T* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream);

// template <typename T>
// void LaunchUpdateKernel(T* log_probs,
//                         T* cum_log_probs,
//                         bool* finished,
//                         int* sequence_length,
//                         int* parent_ids,
//                         int* word_ids,
//                         int* output_ids,
//                         int batch_size,
//                         int num_beams,
//                         int vocab_size,
//                         int eos_token_id,
//                         cudaStream_t stream);

template <typename T>
void LaunchLogitsProcessKernel(
    T* log_probs,
    const int* vocab_mask,
    const int* prefix_vocab_mask,
    int batch_size,
    int num_beams,
    int vocab_size,
    cudaStream_t stream);

// void LaunchInitKernel(
//     half* beam_scores,
//     int batch_size,
//     int num_beams,
//     cudaStream_t stream);

void LaunchInitKernel(
    float* beam_scores,
    int batch_size,
    int num_beams,
    cudaStream_t stream);

void LaunchNextTokenKernel(const int64_t* next_token_indices,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           int batch_size,
                           int top_k,
                           int vocab_size,
                           cudaStream_t stream);

void LaunchUpdateKernel(const int32_t* old_mask_data,
                        int32_t* mask_data,
                        int32_t* next_positions,
                        int batch_beam_size,
                        int current_length,
                        cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
