// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MinMaxScalerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {
template <typename InputType, typename TransformedType>
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& training_batches) {
  using EstimatorT = NS::Featurizers::MinMaxScalerEstimator<InputType, TransformedType>;
  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0);
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, training_batches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, MinMaxScalerTransformer_float) {
  using InputType = float;
  using TransformedType = double;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(static_cast<InputType>(-1)),
      NS::TestHelpers::make_vector<InputType>(static_cast<InputType>(-0.5)),
      NS::TestHelpers::make_vector<InputType>(static_cast<InputType>(0)),
      NS::TestHelpers::make_vector<InputType>(static_cast<InputType>(1)));

  auto stream = GetStream<InputType, TransformedType>(training_batches);

  OpTester test("MinMaxScalerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {1}, {2});
  test.AddOutput<TransformedType>("Output", {1}, {1.5f});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, MinMaxScalerTransformer_only_one_input) {
  using InputType = int8_t;
  using TransformedType = double;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(static_cast<InputType>(-1)));

  auto stream = GetStream<InputType, TransformedType>(training_batches);

  OpTester test("MinMaxScalerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {1}, {2});
  test.AddOutput<TransformedType>("Output", {1}, {0.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime