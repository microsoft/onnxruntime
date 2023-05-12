// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"

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

#ifdef USE_ROCM
  OrtROCMProviderOptions rocm_options;
  session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
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
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
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

TEST(BeamSearchTest, GptBeamSearchWithInitDecoderFp16) {
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
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to True as is the default option)
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx"), session_options);

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
TEST(BeamSearchTest, GptBeamSearchFp16_VocabPadded) {
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
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The following model was obtained by padding the vocabulary size in testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx
    // from 1000 to 1600 (just for illustrative and testing purposes) to see if the beam search implementation can handle
    // such a scenario
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16_padded_vocab.onnx"), session_options);

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

TEST(BeamSearchTest, WhisperBeamSearchFp32) {
  // Load audio input features
  int64_t batch_size = 1;
  const int64_t feature_size = 80;
  const int64_t encoder_sequence_length = 3000;
  std::vector<int64_t> input_features_shape{batch_size, feature_size, encoder_sequence_length};

  std::ifstream input_stream(ORT_TSTR("testdata/whisper_input_features.txt"));
  std::istream_iterator<float> start(input_stream), end;
  std::vector<float> input_features(start, end);

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{26};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{2};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> decoder_input_ids_shape{1, 4};
  std::vector<int32_t> decoder_input_ids{{50258, 50259, 50359, 50363}};

  std::vector<int64_t> expected_output_shape{batch_size, num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{{
      50258, 50259, 50359, 50363, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295,
      264, 2808, 5359, 293, 321, 366, 5404, 281, 2928, 702, 14943, 13, 50257}};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_features_tensor = Ort::Value::CreateTensor(
      info, input_features.data(), input_features.size(), input_features_shape.data(), input_features_shape.size());

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

  auto decoder_input_ids_tensor = Ort::Value::CreateTensor(
      info, decoder_input_ids.data(), decoder_input_ids.size(), decoder_input_ids_shape.data(), decoder_input_ids_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_features_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  ort_inputs.push_back(std::move(decoder_input_ids_tensor));
  const char* input_names[] = {"input_features", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty", "decoder_input_ids"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  // The ONNX model is generated like the following:
  // python convert_to_onnx.py -m openai/whisper-tiny --output whispertiny -e
  Ort::Session session(*ort_env, ORT_TSTR("testdata/whisper-tiny_beamsearch.onnx"), session_options);
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

}  // namespace test
}  // namespace onnxruntime
