#include "contrib_ops/cuda/transformers/beam_search_topk.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <random>

#include <cuda_runtime.h>
#include "gtest/gtest.h"

namespace onnxruntime {
namespace cuda {
namespace test {

void FillAndShuffle(std::vector<float>& values, int32_t batch_size, int32_t beam_size, int32_t vocab_size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  for (int32_t batch = 0; batch < batch_size; batch++) {
    int32_t batch_base_idx = batch * beam_size * vocab_size;
    for (int32_t beam = 0; beam < beam_size; beam++) {
      int32_t value = beam;
      int32_t beam_base_idx = beam * vocab_size;
      for (int32_t vocab = 0; vocab < vocab_size; vocab++) {
        values[batch_base_idx + beam_base_idx + vocab] = (float)(value);
        value += beam_size;
      }
      std::shuffle(values.begin() + batch_base_idx + beam_base_idx,
                   values.begin() + batch_base_idx + beam_base_idx + vocab_size,
                   generator);
    }
  }
}

void ComputeTopKReference(const std::vector<float>& values,
                          std::vector<float>& top_k_values,
                          std::vector<int32_t>& top_k_tokens,
                          std::vector<int32_t>& top_k_indices,
                          int32_t batch_size,
                          int32_t beam_size,
                          int32_t vocab_size,
                          int32_t k) {
  using VK = std::pair<float, int32_t>;

  for (int32_t b = 0; b < batch_size; b++) {
    std::priority_queue<VK, std::vector<VK>, std::greater<VK>> queue;

    int32_t base_idx = b * beam_size * vocab_size;

    // initialize queue with k elements
    for (int32_t i = 0; i < k; i++) {
      queue.push({values[base_idx + i], i});
    }
    for (int32_t i = k; i < beam_size * vocab_size; i++) {
      if (values[base_idx + i] > queue.top().first) {
        queue.pop();
        queue.push({values[base_idx + i], i});
      }
    }

    int32_t top_k_base_idx = b * k;
    for (int32_t i = k - 1; i >= 0; i--) {
      top_k_values[top_k_base_idx + i] = queue.top().first;
      top_k_tokens[top_k_base_idx + i] = queue.top().second % vocab_size;
      top_k_indices[top_k_base_idx + i] = queue.top().second / vocab_size;
      queue.pop();
    }
  }
}

TEST(TestBeamSearch, TopK) {
  int32_t batch_size = 4;
  int32_t beam_size = 4;
  int32_t vocab_size = 50257;
  int32_t k = 2 * beam_size;
  int32_t batch_x_beam_x_vocab = batch_size * beam_size * vocab_size;
  std::vector<float> values(batch_x_beam_x_vocab);
  FillAndShuffle(values, batch_size, beam_size, vocab_size);

  std::vector<float> top_k_values_ref(batch_size * k);
  std::vector<int32_t> top_k_tokens_ref(batch_size * k);
  std::vector<int32_t> top_k_indices_ref(batch_size * k);
  ComputeTopKReference(values, top_k_values_ref, top_k_tokens_ref, top_k_indices_ref, batch_size, beam_size, vocab_size, k);

  const int32_t max_vocab_parts = 128;
  size_t buffer_size = batch_x_beam_x_vocab * 4                                      // input
                       + batch_size * beam_size * k * (max_vocab_parts + 1) * 2 * 4  // tmp
                       + batch_size * k * 3 * 4;                                     // output size
  void* cuda_buffer = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&cuda_buffer, buffer_size));
  float* values_device = (float*)cuda_buffer;
  float* top_k_1st_values_tmp = (float*)(values_device + batch_x_beam_x_vocab);
  int32_t* top_k_1st_tokens_tmp = (int32_t*)(top_k_1st_values_tmp + batch_size * beam_size * k * max_vocab_parts);
  float* top_k_2nd_values_tmp = (float*)(top_k_1st_tokens_tmp + batch_size * beam_size * k * max_vocab_parts);
  int32_t* top_k_2nd_tokens_tmp = (int32_t*)(top_k_2nd_values_tmp + batch_size * beam_size * k);
  float* top_k_value = (float*)(top_k_2nd_tokens_tmp + batch_size * beam_size * k);
  int32_t* top_k_token = (int32_t*)(top_k_value + batch_size * k);
  int32_t* top_k_indices = (int32_t*)(top_k_token + batch_size * k);
  CUDA_CALL_THROW(cudaMemcpy(values_device, values.data(), batch_x_beam_x_vocab * 4, cudaMemcpyHostToDevice));

  contrib::cuda::BeamSearchTopK(values_device,
                                batch_size,
                                beam_size,
                                vocab_size,
                                k,
                                top_k_1st_values_tmp,
                                top_k_1st_tokens_tmp,
                                top_k_2nd_values_tmp,
                                top_k_2nd_tokens_tmp,
                                top_k_value,
                                top_k_token,
                                top_k_indices,
                                NULL /*stream*/);

  std::vector<float> top_k_values_host(batch_size * k);
  std::vector<int32_t> top_k_token_host(batch_size * k);
  std::vector<int32_t> top_k_indices_host(batch_size * k);
  CUDA_CALL_THROW(cudaMemcpy(top_k_values_host.data(), top_k_value, batch_size * k * 4, cudaMemcpyDeviceToHost));
  CUDA_CALL_THROW(cudaMemcpy(top_k_token_host.data(), top_k_token, batch_size * k * 4, cudaMemcpyDeviceToHost));
  CUDA_CALL_THROW(cudaMemcpy(top_k_indices_host.data(), top_k_indices, batch_size * k * 4, cudaMemcpyDeviceToHost));
  for (int32_t i = 0; i < batch_size * k; i++) {
    ASSERT_TRUE(top_k_values_ref[i] == top_k_values_host[i] &&
                top_k_tokens_ref[i] == top_k_token_host[i] &&
                top_k_indices_ref[i] == top_k_indices_host[i]);
  }

  CUDA_CALL_THROW(cudaFree(cuda_buffer));
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
