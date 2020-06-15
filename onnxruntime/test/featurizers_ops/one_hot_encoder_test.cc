// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/OneHotEncoderFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {
template <typename InputType>
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& training_batches, bool allow_missing_values) {
  using Estimator = NS::Featurizers::OneHotEncoderEstimator<InputType>;

  Estimator estimator(NS::CreateTestAnnotationMapsPtr(1), 0, allow_missing_values);
  NS::TestHelpers::Train<Estimator, InputType>(estimator, training_batches);
  typename Estimator::TransformerUniquePtr pTransformer(estimator.create_transformer());

  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, OneHotEncoder_uint32_t) {
  using InputType = uint32_t;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(10, 20, 10),
      NS::TestHelpers::make_vector<InputType>(30),
      NS::TestHelpers::make_vector<InputType>(10, 10, 11, 15),
      NS::TestHelpers::make_vector<InputType>(18, 8));

  auto stream = GetStream<InputType>(training_batches, false);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("OneHotEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {11u, 8u, 10u, 15u, 20u});

  test.AddOutput<uint64_t>("NumElements", {5}, {7u, 7u, 7u, 7u, 7u});
  test.AddOutput<uint8_t>("Value", {5}, {1u, 1u, 1u, 1u, 1u});
  test.AddOutput<uint64_t>("Index", {5}, {2u, 0u, 1u, 3u, 5u});

  test.Run();
}

TEST(FeaturizersTests, OneHotEncoder_string) {
  using InputType = std::string;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("orange", "apple", "orange",
                                              "grape", "carrot", "carrot",
                                              "peach", "banana", "orange"));

  auto stream = GetStream<InputType>(training_batches, false);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("OneHotEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {4}, {"banana", "grape", "apple", "orange"});

  test.AddOutput<uint64_t>("NumElements", {4}, {6u, 6u, 6u, 6u});
  test.AddOutput<uint8_t>("Value", {4}, {1u, 1u, 1u, 1u});
  test.AddOutput<uint64_t>("Index", {4}, {1u, 3u, 0u, 4u});

  test.Run();
}

TEST(FeaturizersTests, OneHotEncoder_unseen_values) {
  using InputType = std::string;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("orange", "apple", "orange",
                                              "grape", "carrot", "carrot",
                                              "peach", "banana", "orange"));

  auto stream = GetStream<InputType>(training_batches, true);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("OneHotEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {"banana", "grape", "apple", "orange", "hello"});

  test.AddOutput<uint64_t>("NumElements", {5}, {7u, 7u, 7u, 7u, 7u});
  test.AddOutput<uint8_t>("Value", {5}, {1u, 1u, 1u, 1u, 1u});
  test.AddOutput<uint64_t>("Index", {5}, {2u, 4u, 1u, 5u, 0u});

  test.Run();
}

TEST(FeaturizersTests, OneHotEncoder_unseen_values_throws) {
  using InputType = std::string;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("orange", "apple", "orange",
                                              "grape", "carrot", "carrot",
                                              "peach", "banana", "orange"));

  auto stream = GetStream<InputType>(training_batches, false);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("OneHotEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {"banana", "grape", "apple", "orange", "hello"});

  test.AddOutput<uint64_t>("NumElements", {5}, {7u, 7u, 7u, 7u, 7u});
  test.AddOutput<uint8_t>("Value", {5}, {1u, 1u, 1u, 1u, 1u});
  test.AddOutput<uint64_t>("Index", {5}, {2u, 4u, 1u, 5u, 0u});

  test.Run(OpTester::ExpectResult::kExpectFailure);
}

}  // namespace test
}  // namespace onnxruntime
