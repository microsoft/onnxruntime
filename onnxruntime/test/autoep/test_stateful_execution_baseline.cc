// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

class StatefulExecutionBaselineTest : public testing::TestWithParam<bool> {};

Ort::Session CreateReferenceSession(const ONNX_NAMESPACE::ModelProto& model, bool use_cuda) {
  const std::string bytes = model.SerializeAsString();
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  if (use_cuda) {
    options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  }
  Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
  const auto subgraphs = session.GetEpGraphAssignmentInfo();
  EXPECT_FALSE(subgraphs.empty());
  for (const auto& subgraph : subgraphs) {
    EXPECT_EQ(subgraph.GetEpName(), use_cuda ? "CUDAExecutionProvider" : "CPUExecutionProvider");
  }
  return session;
}

Ort::Value MakeScalar(float& value) {
  return Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault), &value, 1, nullptr, 0);
}

TEST_P(StatefulExecutionBaselineTest, SessionDoesNotRetainFeedback) {
  auto session = CreateReferenceSession(stateful::MakeFeedbackModel(), GetParam());
  constexpr std::array<const char*, 2> input_names{"input_value", "past_value"};
  constexpr std::array<const char*, 2> output_names{"present_value", "done"};
  float value = 2.0f;
  float past = 0.0f;
  std::array<Ort::Value, 2> inputs{MakeScalar(value), MakeScalar(past)};

  auto first = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                           inputs.size(), output_names.data(), output_names.size());
  auto repeated = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                              inputs.size(), output_names.data(), output_names.size());
  EXPECT_EQ(*first[0].GetTensorData<float>(), 2.0f);
  EXPECT_EQ(*repeated[0].GetTensorData<float>(), 2.0f);
  EXPECT_FALSE(*repeated[1].GetTensorData<bool>());

  inputs[1] = std::move(first[0]);
  auto feedback = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                              inputs.size(), output_names.data(), output_names.size());
  EXPECT_EQ(*feedback[0].GetTensorData<float>(), 4.0f);
  EXPECT_EQ(*repeated[0].GetTensorData<float>(), 2.0f);
}

TEST_P(StatefulExecutionBaselineTest, FiniteInputsAndBooleanOutputHaveKnownReference) {
  auto session = CreateReferenceSession(stateful::MakeFeedbackModel(), GetParam());
  constexpr std::array<const char*, 2> input_names{"input_value", "past_value"};
  constexpr std::array<const char*, 2> output_names{"present_value", "done"};
  constexpr std::array<float, 3> values{2.0f, 3.0f, 4.0f};
  constexpr std::array<float, 3> expected_values{2.0f, 5.0f, 9.0f};
  constexpr std::array<bool, 3> expected_done{false, false, true};
  float past = 0.0f;

  for (size_t iteration = 0; iteration < values.size(); ++iteration) {
    float value = values[iteration];
    std::array<Ort::Value, 2> inputs{MakeScalar(value), MakeScalar(past)};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                               inputs.size(), output_names.data(), output_names.size());
    EXPECT_EQ(*outputs[0].GetTensorData<float>(), expected_values[iteration]);
    EXPECT_EQ(*outputs[1].GetTensorData<bool>(), expected_done[iteration]);
    past = *outputs[0].GetTensorData<float>();
  }
}

TEST_P(StatefulExecutionBaselineTest, RenamedStagesHaveKnownReference) {
  auto first = CreateReferenceSession(stateful::MakeFeedbackModel(), GetParam());
  auto second = CreateReferenceSession(stateful::MakeRenamedStageModel(), GetParam());
  constexpr std::array<const char*, 2> input_names{"input_value", "past_value"};
  constexpr std::array<const char*, 1> output_names{"present_value"};
  float value = 3.0f;
  float past = 1.0f;
  std::array<Ort::Value, 2> inputs{MakeScalar(value), MakeScalar(past)};
  auto first_outputs = first.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                                 inputs.size(), output_names.data(), output_names.size());

  constexpr std::array<const char*, 1> next_input_names{"upstream"};
  constexpr std::array<const char*, 1> next_output_names{"pipeline_result"};
  auto second_outputs = second.Run(Ort::RunOptions{nullptr}, next_input_names.data(), first_outputs.data(),
                                   first_outputs.size(), next_output_names.data(), next_output_names.size());
  EXPECT_EQ(*first_outputs[0].GetTensorData<float>(), 4.0f);
  EXPECT_EQ(*second_outputs[0].GetTensorData<float>(), 8.0f);
}

TEST_P(StatefulExecutionBaselineTest, DecoderLogitsAndGrowingCacheHaveKnownReference) {
  auto session = CreateReferenceSession(stateful::MakeDecoderModel(), GetParam());
  const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  constexpr std::array<const char*, 2> input_names{"token", "past_key"};
  constexpr std::array<const char*, 2> output_names{"logits", "present_key"};
  constexpr std::array<int64_t, 1> token_shape{1};
  constexpr std::array<int64_t, 4> empty_cache_shape{1, 1, 0, 1};
  constexpr std::array<std::array<float, 4>, 3> expected_logits{{
      {0.f, 4.f, 1.f, -2.f},
      {0.f, 1.f, 5.f, -2.f},
      {-1.f, 1.f, 2.f, 6.f},
  }};
  float unused = 0.0f;
  auto cache = Ort::Value::CreateTensor<float>(
      memory_info, &unused, 0, empty_cache_shape.data(), empty_cache_shape.size());
  Ort::Value retained_cache{nullptr};

  for (int64_t token = 0; token < 3; ++token) {
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<int64_t>(memory_info, &token, 1, token_shape.data(), token_shape.size()),
        std::move(cache)};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                               inputs.size(), output_names.data(), output_names.size());
    const auto* logits = outputs[0].GetTensorData<float>();
    for (size_t index = 0; index < 4; ++index) {
      EXPECT_EQ(logits[index], expected_logits[token][index]);
    }
    const auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    ASSERT_EQ(shape, (std::vector<int64_t>{1, 1, token + 1, 1}));
    const auto* keys = outputs[1].GetTensorData<float>();
    for (int64_t index = 0; index <= token; ++index) {
      EXPECT_EQ(keys[index], static_cast<float>(index));
    }
    if (token == 0) {
      retained_cache = std::move(outputs[1]);
      cache = Ort::Value::CreateTensor<float>(
          memory_info, retained_cache.GetTensorMutableData<float>(), 1, shape.data(), shape.size());
    } else {
      cache = std::move(outputs[1]);
    }
  }

  ASSERT_EQ(retained_cache.GetTensorTypeAndShapeInfo().GetElementCount(), 1u);
  EXPECT_EQ(*retained_cache.GetTensorData<float>(), 0.0f);
}

TEST_P(StatefulExecutionBaselineTest, DenoiserTimestepBroadcastHasKnownReference) {
  auto session = CreateReferenceSession(stateful::MakeDenoiserModel(), GetParam());
  constexpr std::array<const char*, 2> input_names{"latent", "timestep"};
  constexpr std::array<const char*, 1> output_names{"noise_prediction"};
  constexpr std::array<int64_t, 4> shape{1, 1, 2, 2};
  std::array<float, 4> latent{1.0f, -2.0f, 3.0f, -4.0f};
  float timestep = 2.0f;
  std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
          latent.data(), latent.size(), shape.data(), shape.size()),
      MakeScalar(timestep)};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                             inputs.size(), output_names.data(), output_names.size());
  const auto* prediction = outputs[0].GetTensorData<float>();
  constexpr std::array<float, 4> expected{3.0f, 0.0f, 5.0f, -2.0f};
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(prediction[index], expected[index]);
  }
}

INSTANTIATE_TEST_SUITE_P(Cpu, StatefulExecutionBaselineTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, StatefulExecutionBaselineTest, testing::Values(true));
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
