// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"
#include "contrib_ops/cuda/transformers/beam_search_topk.h"
#include <random>
#include <numeric>
#include <queue>
#include <algorithm>
#include <cuda_runtime.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(BeamSearchTest, GptBeamSearchFp32) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx"), session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}

TEST(BeamSearchTest, GptBeamSearchFp16) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  if (HasCudaEnvironment(min_cuda_architecture)) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_fp16.onnx  -p fp16 --use_gpu --max_length 20
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}

#if 0
void FillAndShuffle(std::vector<float>& values, int32_t parts, int32_t vocab_size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  for (int32_t i = 0; i < parts; i++) {
    for (int32_t v = 0; v < vocab_size; v++) {
      values[i * vocab_size + v] = float(v);
    }
    std::shuffle(values.begin() + i * vocab_size, values.end() + (i + 1) * vocab_size, generator);
  }
}

void TopKReference(const std::vector<float>& values,
          std::vector<float>& top_k_values,
          std::vector<int32_t>& top_k_indices,
          int32_t parts,
          int32_t vocab_size,
          int32_t k) {
  auto value_compare = [&values](const int32_t idx_1, const int32_t idx_2) { return values[idx_1] > values[idx_2]; };

  using VK = std::pair<float, int32_t>;

  for (int32_t p = 0; p < parts; p++) {
    std::priority_queue < VK, std::vector<VK>, std::greater<VK>> queue;

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

TEST(BeamSearchTest, TopK) {
  int32_t batch_size = 4;
  int32_t beam_size = 4;
  int32_t vocab_size = 50257;
  int32_t k = 2 * beam_size;
  std::vector<float> values(batch_size * beam_size * vocab_size);
  FillAndShuffle(values, batch_size * beam_size, vocab_size);

  std::vector<float> top_k_values_ref(batch_size * beam_size * k);
  std::vector<int32_t> top_k_indices_ref(batch_size * beam_size * k);
  TopKReference(values, top_k_values_ref, top_k_indices_ref, batch_size * beam_size, vocab_size, k);

  const int32_t max_vocab_parts = 128;
  void* cuda_buffer = nullptr;
  cudaMalloc(&cuda_buffer, static_cast<size_t>(batch_size * beam_size * vocab_size * 4 + batch_size * beam_size * k * (max_vocab_parts + 2) * 8));
  float* values_device_device = (float*)cuda_buffer;
  float* top_k_values_device = values_device_device + batch_size * beam_size * vocab_size;
  int32_t* output_indices_device = (int32_t*)(top_k_values_device + batch_size * beam_size * k);
  float* output_values_device_tmp = (float*)(output_indices_device + batch_size * beam_size * k);
  int32_t* output_indices_device_tmp = (int32_t*)(output_values_device_tmp + batch_size * beam_size * k * max_vocab_parts);
  cudaMemcpy(values_device_device, values.data(), batch_size * beam_size * vocab_size * 4, cudaMemcpyHostToDevice);
  contrib::cuda::LaunchTopK(
      values_device_device,
      batch_size,
      beam_size,
      vocab_size,
      k,
      top_k_values_device,
      output_indices_device,
      output_values_device_tmp,
      output_indices_device_tmp,
      NULL /*stream*/);

  std::vector<float> top_k_values_host(batch_size * beam_size * k);
  std::vector<int32_t> top_k_indices_host(batch_size * beam_size * k);
  cudaMemcpy(top_k_values_host.data(), top_k_values_device, batch_size * beam_size * k * 4, cudaMemcpyDeviceToHost);
  for (int32_t i = 0; i < batch_size * beam_size * k; i++) {
    EXPECT_EQ(top_k_values_ref[i], top_k_values_host[i]);
    EXPECT_EQ(top_k_indices_ref[i], top_k_indices_host[i]);
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
