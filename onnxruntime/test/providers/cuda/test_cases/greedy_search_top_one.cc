
#include "contrib_ops/cuda/transformers/greedy_search_top_one.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#include <algorithm>
#include <numeric>
#include <random>

#include <cuda_runtime.h>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace cuda {
namespace test {

void FillAndShuffle(std::vector<float>& values, int32_t batch_size, int32_t vocab_size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  for (int32_t batch = 0; batch < batch_size; batch++) {
    for (int32_t vocab = 0; vocab < vocab_size; vocab++) {
      values[batch * vocab_size + vocab] = static_cast<float>(vocab);
    }
    std::shuffle(values.begin() + batch * vocab_size,
                 values.begin() + batch * vocab_size + vocab_size,
                 generator);
  }
}

void ComputeTop1Reference(const std::vector<float>& values,
                          std::vector<float>& top_k_values,
                          std::vector<int32_t>& top_k_tokens,
                          int32_t batch_size,
                          int32_t vocab_size) {
  for (int32_t b = 0; b < batch_size; b++) {
    int32_t base_idx = b * vocab_size;

    auto max_itr = std::max_element(values.begin() + base_idx, values.begin() + base_idx + vocab_size);
    top_k_values[b] = *max_itr;
    top_k_tokens[b] = static_cast<int32_t>(std::distance(values.begin() + base_idx, max_itr));
  }
}

TEST(TestGreedySearch, TopOne) {
  int32_t batch_size = 4;
  int32_t vocab_size = 50257;
  int32_t batch_x_vocab = batch_size * vocab_size;
  std::vector<float> values(batch_x_vocab);
  FillAndShuffle(values, batch_size, vocab_size);

  std::vector<float> top_k_values_ref(batch_size);
  std::vector<int32_t> top_k_tokens_ref(batch_size);
  ComputeTop1Reference(values, top_k_values_ref, top_k_tokens_ref, batch_size, vocab_size);

  constexpr size_t kMaxPartsPerVocab = 128;
  const size_t stage_1_element_size = kMaxPartsPerVocab * batch_size;
  const size_t output_element_size = batch_size;

  size_t input_buffer_size = batch_x_vocab * sizeof(float);
  size_t top1_temp_buffer = stage_1_element_size * (sizeof(float) + sizeof(int32_t));
  size_t output_buffer = output_element_size * (sizeof(float) + sizeof(int32_t));
  size_t buffer_size = input_buffer_size + top1_temp_buffer + output_buffer;

  void* topk_data = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&topk_data, buffer_size));

  float* input_data_gpu = (float*)topk_data;
  float* stage_1_scores_data = reinterpret_cast<float*>(input_data_gpu + batch_x_vocab);
  int32_t* stage_1_token_data = reinterpret_cast<int32_t*>(stage_1_scores_data + stage_1_element_size);

  float* output_score_data = reinterpret_cast<float*>(stage_1_token_data + stage_1_element_size);
  int32_t* output_token_data = reinterpret_cast<int32_t*>(output_score_data + output_element_size);

  CUDA_CALL_THROW(cudaMemcpy(input_data_gpu, values.data(), input_buffer_size, cudaMemcpyHostToDevice));

  contrib::cuda::GreedySearchTopOne(
      input_data_gpu,
      batch_size,
      vocab_size,
      stage_1_scores_data,
      stage_1_token_data,
      output_score_data,
      output_token_data,
      NULL /*stream*/);

  std::vector<float> top_k_values_host(batch_size);
  std::vector<int32_t> top_k_token_host(batch_size);
  CUDA_CALL_THROW(cudaMemcpy(top_k_values_host.data(), output_score_data, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL_THROW(cudaMemcpy(top_k_token_host.data(), output_token_data, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
  for (int32_t i = 0; i < batch_size; i++) {
    ASSERT_TRUE(top_k_values_ref[i] == top_k_values_host[i] &&
                top_k_tokens_ref[i] == top_k_token_host[i]);
  }

  CUDA_CALL_THROW(cudaFree(topk_data));
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
