#ifndef NDEBUG

#include "contrib_ops/cuda/transformers/beam_search_topk.h"
#include "core/providers/cuda/math/topk_impl.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <random>

#include <cuda_runtime.h>

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
                          int32_t padded_vocab_size,
                          int32_t k) {
  using VK = std::pair<float, int32_t>;

  for (int32_t b = 0; b < batch_size; b++) {
    std::priority_queue<VK, std::vector<VK>, std::greater<VK>> queue;

    int32_t base_idx = b * beam_size * padded_vocab_size;

    // initialize queue with k elements
    for (int32_t i = 0; i < k; i++) {
      queue.push({values[base_idx + i], i});
    }

    // We want to ignore any padding region (if any), hence
    // only iterate till `beam_size * vocab_size`
    int32_t iter = k;
    int32_t num_elements_seen_in_beam = k;
    while (iter < beam_size * padded_vocab_size) {
      // If we reach the end of this beam (i.e.) vocab_size,
      // we skip over any padded region
      if (num_elements_seen_in_beam == vocab_size) {
        iter += (padded_vocab_size - vocab_size);
        num_elements_seen_in_beam = 0;
        continue;
      }

      if (values[base_idx + iter] > queue.top().first) {
        queue.pop();
        queue.push({values[base_idx + iter], iter});
      }

      ++iter;
      ++num_elements_seen_in_beam;
    }

    int32_t top_k_base_idx = b * k;
    for (int32_t i = k - 1; i >= 0; i--) {
      top_k_values[top_k_base_idx + i] = queue.top().first;

      // Account for any padding
      top_k_tokens[top_k_base_idx + i] = queue.top().second % padded_vocab_size;
      top_k_indices[top_k_base_idx + i] = queue.top().second / padded_vocab_size;
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
  FillAndShuffle(values, batch_size, beam_size, vocab_size);

  std::vector<float> top_k_values_ref(batch_size * k);
  std::vector<int32_t> top_k_tokens_ref(batch_size * k);
  std::vector<int32_t> top_k_indices_ref(batch_size * k);
  ComputeTopKReference(values, top_k_values_ref, top_k_tokens_ref,
                       top_k_indices_ref, batch_size, beam_size,
                       vocab_size, /*no padding*/ vocab_size, k);

  const int32_t max_vocab_parts = 128;
  size_t buffer_size = batch_x_beam_x_vocab * 4                                      // input
                       + batch_size * beam_size * k * (max_vocab_parts + 1) * 2 * 4  // tmp
                       + batch_size * k * 3 * 4;                                     // output size

  void* cuda_buffer = nullptr;
  cudaMalloc(&cuda_buffer, buffer_size);

  // Define deleter for the cuda memory
  auto cuda_deleter = [](float* ptr) -> void {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  };

  std::unique_ptr<float, decltype(cuda_deleter)> cuda_buffer_unique_ptr(reinterpret_cast<float*>(cuda_buffer), cuda_deleter);
  (void)(cuda_buffer_unique_ptr);

  float* values_device = (float*)cuda_buffer;
  float* top_k_1st_values_tmp = (float*)(values_device + batch_x_beam_x_vocab);
  int32_t* top_k_1st_tokens_tmp = (int32_t*)(top_k_1st_values_tmp + batch_size * beam_size * k * max_vocab_parts);
  float* top_k_2nd_values_tmp = (float*)(top_k_1st_tokens_tmp + batch_size * beam_size * k * max_vocab_parts);
  int32_t* top_k_2nd_tokens_tmp = (int32_t*)(top_k_2nd_values_tmp + batch_size * beam_size * k);
  float* top_k_value = (float*)(top_k_2nd_tokens_tmp + batch_size * beam_size * k);
  int32_t* top_k_token = (int32_t*)(top_k_value + batch_size * k);
  int32_t* top_k_indices = (int32_t*)(top_k_token + batch_size * k);
  cudaMemcpy(values_device, values.data(), batch_x_beam_x_vocab * 4, cudaMemcpyHostToDevice);

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
  cudaMemcpy(top_k_values_host.data(), top_k_value, batch_size * k * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(top_k_token_host.data(), top_k_token, batch_size * k * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(top_k_indices_host.data(), top_k_indices, batch_size * k * 4, cudaMemcpyDeviceToHost);
  for (int32_t i = 0; i < batch_size * k; i++) {
    if (top_k_values_ref[i] != top_k_values_host[i] ||
        top_k_tokens_ref[i] != top_k_token_host[i] ||
        top_k_indices_ref[i] != top_k_indices_host[i]) {
      return false;
    }
  }

  return true;
}

bool TestGenerationDeviceHelperTopK() {
  // Test RadixTopK path
  bool result_1 = false;
  {
    int64_t batch_size = 8;
    int64_t beam_size = 1;
    // Choose this to be greater than GridDim::maxThreadsPerBlock so that the RadixTopK is chosen
    int64_t vocab_size = 50257;
    int64_t padded_vocab_size = 50264;

    int64_t k = 2 * beam_size;

    int64_t batch_x_beam_x_padded_vocab = batch_size * beam_size * padded_vocab_size;
    std::vector<float> values(batch_x_beam_x_padded_vocab);
    FillAndShuffle(values, batch_size, beam_size, padded_vocab_size);

    std::vector<float> top_k_values_ref(batch_size * k);
    std::vector<int32_t> top_k_tokens_ref(batch_size * k);
    std::vector<int32_t> top_k_indices_ref(batch_size * k);
    ComputeTopKReference(values, top_k_values_ref, top_k_tokens_ref,
                         top_k_indices_ref, batch_size, beam_size,
                         vocab_size, /*padding present*/ padded_vocab_size, k);

    size_t buffer_size = batch_x_beam_x_padded_vocab * 4  // input scores
                         + batch_size * k * 4             // output topk values
                         + batch_size * k * 8;            // first topk output indices

    void* cuda_buffer = nullptr;
    cudaMalloc(&cuda_buffer, buffer_size);

    // Define deleter for the cuda memory
    auto cuda_deleter = [](float* ptr) -> void {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    };

    std::unique_ptr<float, decltype(cuda_deleter)> cuda_buffer_unique_ptr(reinterpret_cast<float*>(cuda_buffer), cuda_deleter);
    (void)(cuda_buffer_unique_ptr);

    float* input_scores = reinterpret_cast<float*>(cuda_buffer);
    cudaMemcpy(input_scores, values.data(), batch_x_beam_x_padded_vocab * 4, cudaMemcpyHostToDevice);

    float* topk_scores = reinterpret_cast<float*>(cuda_buffer) + batch_x_beam_x_padded_vocab;
    int64_t* topk_tokens = reinterpret_cast<int64_t*>(reinterpret_cast<float*>(cuda_buffer) + batch_x_beam_x_padded_vocab + batch_size * k);

    TArray<int64_t> elem_nums_cuda({batch_size * beam_size, padded_vocab_size});
    for (int32_t i = elem_nums_cuda.Size() - 2; i >= 0; --i) {
      elem_nums_cuda[i] *= elem_nums_cuda[i + 1];
    }

    int64_t N = elem_nums_cuda[0] / padded_vocab_size;

    TopKImpl<float>(nullptr,  // We limit number of beams in BeamSearchParameters, so K <= 256 and use NULL here
                    reinterpret_cast<cudaStream_t>(NULL),
                    input_scores,
                    topk_scores,
                    topk_tokens,
                    elem_nums_cuda,
                    static_cast<size_t>(elem_nums_cuda.Size()),
                    static_cast<int32_t>(1),
                    static_cast<int64_t>(k),
                    static_cast<int64_t>(1),
                    static_cast<int64_t>(1),
                    N,
                    vocab_size,
                    &padded_vocab_size);

    std::vector<float> top_k_values_host(batch_size * k);
    std::vector<int64_t> top_k_tokens_host(batch_size * k);
    cudaMemcpy(top_k_values_host.data(), topk_scores, batch_size * k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(top_k_tokens_host.data(), topk_tokens, batch_size * k * 8, cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < batch_size * k; i++) {
      if (top_k_values_ref[i] != top_k_values_host[i] ||
          top_k_tokens_ref[i] != top_k_tokens_host[i]) {
        return false;
      }
    }

    result_1 = true;
  }

  // Test BitonicTopK path
  bool result_2 = false;
  {
    int64_t batch_size = 8;
    int64_t beam_size = 1;

    // Choose this to be less than GridDim::maxThreadsPerBlock so that the BitonicTopK is chosen
    int64_t vocab_size = 7;
    int64_t padded_vocab_size = 8;

    int64_t k = 2 * beam_size;

    int64_t batch_x_beam_x_padded_vocab = batch_size * beam_size * padded_vocab_size;
    std::vector<float> values(batch_x_beam_x_padded_vocab);
    FillAndShuffle(values, batch_size, beam_size, padded_vocab_size);

    std::vector<float> top_k_values_ref(batch_size * k);
    std::vector<int32_t> top_k_tokens_ref(batch_size * k);
    std::vector<int32_t> top_k_indices_ref(batch_size * k);
    ComputeTopKReference(values, top_k_values_ref, top_k_tokens_ref,
                         top_k_indices_ref, batch_size, beam_size,
                         vocab_size, /*padding present*/ padded_vocab_size, k);

    size_t buffer_size = batch_x_beam_x_padded_vocab * 4  // input scores
                         + batch_size * k * 4             // output topk values
                         + batch_size * k * 8;            // first topk output indices

    void* cuda_buffer = nullptr;
    cudaMalloc(&cuda_buffer, buffer_size);

    // Define deleter for the cuda memory
    auto cuda_deleter = [](float* ptr) -> void {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    };

    std::unique_ptr<float, decltype(cuda_deleter)> cuda_buffer_unique_ptr(reinterpret_cast<float*>(cuda_buffer), cuda_deleter);
    (void)(cuda_buffer_unique_ptr);

    float* input_scores = reinterpret_cast<float*>(cuda_buffer);
    cudaMemcpy(input_scores, values.data(), batch_x_beam_x_padded_vocab * 4, cudaMemcpyHostToDevice);

    float* topk_scores = reinterpret_cast<float*>(cuda_buffer) + batch_x_beam_x_padded_vocab;
    int64_t* topk_tokens = reinterpret_cast<int64_t*>(reinterpret_cast<float*>(cuda_buffer) + batch_x_beam_x_padded_vocab + batch_size * k);

    TArray<int64_t> elem_nums_cuda({batch_size * beam_size, padded_vocab_size});
    for (int32_t i = elem_nums_cuda.Size() - 2; i >= 0; --i) {
      elem_nums_cuda[i] *= elem_nums_cuda[i + 1];
    }

    int64_t N = elem_nums_cuda[0] / padded_vocab_size;

    TopKImpl<float>(nullptr,  // We limit number of beams in BeamSearchParameters, so K <= 256 and use NULL here
                    reinterpret_cast<cudaStream_t>(NULL),
                    input_scores,
                    topk_scores,
                    topk_tokens,
                    elem_nums_cuda,
                    static_cast<size_t>(elem_nums_cuda.Size()),
                    static_cast<int32_t>(1),
                    static_cast<int64_t>(k),
                    static_cast<int64_t>(1),
                    static_cast<int64_t>(1),
                    N,
                    vocab_size,
                    &padded_vocab_size);

    std::vector<float> top_k_values_host(batch_size * k);
    std::vector<int64_t> top_k_tokens_host(batch_size * k);
    cudaMemcpy(top_k_values_host.data(), topk_scores, batch_size * k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(top_k_tokens_host.data(), topk_tokens, batch_size * k * 8, cudaMemcpyDeviceToHost);
    for (int32_t i = 0; i < batch_size * k; i++) {
      if (top_k_values_ref[i] != top_k_values_host[i] ||
          top_k_tokens_ref[i] != top_k_tokens_host[i]) {
        return false;
      }
    }

    result_2 = true;
  }

  return result_1 && result_2;
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
#endif
