#ifndef NDEBUG

#include "contrib_ops/cuda/transformers/beam_search_topk.h"
#include <random>
#include <numeric>
#include <queue>
#include <algorithm>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {
namespace test {

void FillAndShuffle(std::vector<float>& values, int32_t parts, int32_t vocab_size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  for (int32_t i = 0; i < parts; i++) {
    for (int32_t v = 0; v < vocab_size; v++) {
      values[i * vocab_size + v] = float(v);
    }
    std::shuffle(values.begin() + i * vocab_size, values.begin() + (i + 1) * vocab_size, generator);
  }
}

void ComputeTopKReference(const std::vector<float>& values,
                          std::vector<float>& top_k_values,
                          std::vector<int32_t>& top_k_indices,
                          int32_t parts,
                          int32_t vocab_size,
                          int32_t k) {
  auto value_compare = [&values](const int32_t idx_1, const int32_t idx_2) { return values[idx_1] > values[idx_2]; };

  using VK = std::pair<float, int32_t>;

  for (int32_t p = 0; p < parts; p++) {
    std::priority_queue<VK, std::vector<VK>, std::greater<VK>> queue;

    int32_t base_idx = p * vocab_size;

    // initialize queue with k elements
    for (int32_t i = 0; i < k; i++) {
      queue.push({values[base_idx + i], i});
    }
    for (int32_t i = k; i < vocab_size; i++) {
      if (values[base_idx + i] > queue.top().first) {
        queue.pop();
        queue.push({values[base_idx + i], i});
      }
    }

    int32_t top_k_base_idx = p * k;
    for (int32_t i = k - 1; i >= 0; i--) {
      top_k_indices[top_k_base_idx + i] = queue.top().second;
      top_k_values[top_k_base_idx + i] = queue.top().first;
      queue.pop();
    }
  }
}

bool TestBeamSearchTopK() {
  int32_t batch_size = 4;
  int32_t beam_size = 4;
  int32_t vocab_size = 50257;
  int32_t k = 2 * beam_size;
  int32_t batch_x_beam_x_vocab = batch_size * beam_size * vocab_size;
  std::vector<float> values(batch_x_beam_x_vocab);
  FillAndShuffle(values, batch_size * beam_size, vocab_size);

  std::vector<float> top_k_values_ref(batch_size * beam_size * k);
  std::vector<int32_t> top_k_indices_ref(batch_size * beam_size * k);
  ComputeTopKReference(values, top_k_values_ref, top_k_indices_ref, batch_size * beam_size, vocab_size, k);

  const int32_t max_vocab_parts = 128;
  void* cuda_buffer = nullptr;
  cudaMalloc(&cuda_buffer, static_cast<size_t>(batch_x_beam_x_vocab * 4 + batch_size * beam_size * k * (max_vocab_parts + 2) * 8));
  float* values_device_device = (float*)cuda_buffer;
  float* top_k_values_device = values_device_device + batch_x_beam_x_vocab;
  int32_t* top_k_indices_device = (int32_t*)(top_k_values_device + batch_size * beam_size * k);
  float* output_values_device_tmp = (float*)(top_k_indices_device + batch_size * beam_size * k);
  int32_t* output_indices_device_tmp = (int32_t*)(output_values_device_tmp + batch_size * beam_size * k * max_vocab_parts);
  cudaMemcpy(values_device_device, values.data(), batch_x_beam_x_vocab * 4, cudaMemcpyHostToDevice);
  contrib::cuda::LaunchTopK(
      values_device_device,
      batch_size,
      beam_size,
      vocab_size,
      k,
      top_k_values_device,
      top_k_indices_device,
      output_values_device_tmp,
      output_indices_device_tmp,
      NULL /*stream*/);

  std::vector<float> top_k_values_host(batch_size * beam_size * k);
  std::vector<int32_t> top_k_indices_host(batch_size * beam_size * k);
  cudaMemcpy(top_k_values_host.data(), top_k_values_device, batch_size * beam_size * k * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(top_k_indices_host.data(), top_k_indices_device, batch_size * beam_size * k * 4, cudaMemcpyDeviceToHost);
  for (int32_t i = 0; i < batch_size * beam_size * k; i++) {
    if (top_k_values_ref[i] != top_k_values_host[i] || top_k_indices_ref[i] != top_k_indices_host[i]) {
      return false;
    }
  }

  return true;
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
#endif