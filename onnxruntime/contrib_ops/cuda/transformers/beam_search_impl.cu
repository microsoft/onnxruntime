// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "beam_search_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__ void NextTokenKernel(const int64_t* next_token_indices,
                                int64_t* next_indices,
                                int64_t* next_tokens,
                                int vocab_size,
                                int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    next_indices[index] = next_token_indices[index] / vocab_size;
    next_tokens[index] = next_token_indices[index] % vocab_size;
  }
}

/* NextToken kernel is corresponding to logic like the following:
   for i in range (batch_size):
    for j in range (top_k):
      next_indices[i, j] = next_token_indices[i, j] / vocab_size
      next_tokens[i, j] = next_token_indices[i, j] % vocab_size
*/
void LaunchNextTokenKernel(const int64_t* next_token_indices,
                           int64_t* next_indices,
                           int64_t* next_tokens,
                           int batch_size,
                           int top_k,
                           int vocab_size,
                           cudaStream_t stream) {
  int total_elements = batch_size * top_k;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  NextTokenKernel<<<gridSize, blockSize, 0, stream>>>(next_token_indices, next_indices, next_tokens, vocab_size, total_elements);
}

__global__ void InitKernel(float* beam_scores,
                           int num_beams,
                           int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    int beam_index = index % num_beams;
    beam_scores[index] = beam_index > 0 ? static_cast<float>(-1e9) : 0.0f;  // This value exceeds limit of MLFloat16 so it is for float only.
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

template <typename T>
__global__ void VocabMaskKernel(T* log_probs,
                                const int* vocab_mask,
                                int vocab_size,
                                int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int word_id = index % vocab_size;

  if (index < total_elements && vocab_mask[word_id] == 0) {
    log_probs[index] = std::numeric_limits<T>::lowest();
  }
}

template <typename T>
void LaunchVocabMaskKernel(
    T* log_probs,
    const int* vocab_mask,
    int batch_size,
    int num_beams,
    int vocab_size,
    cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  VocabMaskKernel<float><<<gridSize, blockSize, 0, stream>>>(log_probs, vocab_mask, vocab_size, total_elements);
}

// Instantiation
template void LaunchVocabMaskKernel(
    float* log_probs,
    const int* vocab_mask,
    int batch_size,
    int num_beams,
    int vocab_size,
    cudaStream_t stream);

template <typename T>
__global__ void AddProbsKernel(T* log_probs,
                               T* cum_log_probs,
                               const int vocab_size,
                               const int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_beam_index = index / vocab_size;

  if (index < total_elements)
    log_probs[index] += cum_log_probs[batch_beam_index];
}

void LaunchAddProbsKernel(float* log_probs,
                          float* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  AddProbsKernel<float><<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}

template <typename T>
__global__ void UpdateInputsKernel(const T* old_mask_data,
                                   T* mask_data,
                                   int64_t* next_positions,
                                   int batch_beam_size,
                                   int current_length) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < batch_beam_size * current_length) {
    // Update attention mask like the following:
    //   for (int i = 0; i < batch_beam_size; i++) {
    //     for (int j = 0; j < current_length - 1; j++) {
    //       mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    //     }
    //     mask_data[i * current_length + current_length - 1] = 1.0f;
    //   }
    int i = index / current_length;
    int j = index % current_length;
    mask_data[index] = (j < current_length - 1) ? old_mask_data[i * (current_length - 1) + j] : static_cast<T>(1.0f);

    // Update sequence length (or next positions) like the following:
    //   for (int i = 0; i < batch_beam_size; i++) {
    //     next_positions[i]++;
    //   }
    if (index < batch_beam_size) {
      next_positions[index]++;
    }
  }
}

template <typename T>
void LaunchUpdateKernel(const T* old_mask_data,
                        T* mask_data,
                        int64_t* next_positions,
                        int batch_beam_size,
                        int current_length,
                        cudaStream_t stream) {
  assert(current_length > 0);
  int total_elements = batch_beam_size * current_length;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  UpdateInputsKernel<T><<<gridSize, blockSize, 0, stream>>>(old_mask_data, mask_data, next_positions, batch_beam_size, current_length);
}

// Instantiation
template void LaunchUpdateKernel(const float* old_mask_data,
                                 float* mask_data,
                                 int64_t* next_positions,
                                 int batch_beam_size,
                                 int current_length,
                                 cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime