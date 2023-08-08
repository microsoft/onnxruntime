// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(GreedySearchTest, GptGreedySearchFp16_VocabPadded) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{
      0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{10};
  std::vector<int32_t> min_length{1};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  if (HasCudaEnvironment(min_cuda_architecture)) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

    // The following model was obtained by padding the vocabulary size in testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx
    // (by making beam_size == 1) from 1000 to 1600 (just for illustrative and testing purposes) to see if the greedy search
    // implementation can handle such a scenario.
    // Check beam_search_test.cc to see how tiny_gpt2_beamsearch_fp16.onnx was generated.
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_greedysearch_fp16_padded_vocab.onnx"), session_options);

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

TEST(GreedySearchTest, GptGreedySearchFp32) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{
      0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{10};
  std::vector<int32_t> min_length{1};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  if (HasCudaEnvironment(min_cuda_architecture)) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_greedysearch_with_init_decoder.onnx"), session_options);

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

}  // namespace test
}  // namespace onnxruntime
