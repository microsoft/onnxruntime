// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/RobustScalerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

template <typename InputT, typename TransformedT>
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputT>>& training_batches, bool centering) {
  using EstimatorT = NS::Featurizers::RobustScalerEstimator<InputT, TransformedT>;
  auto estimator = EstimatorT::CreateWithDefaultScaling(NS::CreateTestAnnotationMapsPtr(1), 0, centering);
  NS::TestHelpers::Train<EstimatorT, InputT>(estimator, training_batches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

uint8_t operator"" _ui8(unsigned long long v) {
  return static_cast<uint8_t>(v);
}

}  // namespace

TEST(FeaturizersTests, RobustScalerTransformer_input_int8__output_float_centering) {
  using InputType = uint8_t;
  using TransformedType = float;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(1_ui8),
      NS::TestHelpers::make_vector<InputType>(7_ui8),
      NS::TestHelpers::make_vector<InputType>(5_ui8),
      NS::TestHelpers::make_vector<InputType>(3_ui8),
      NS::TestHelpers::make_vector<InputType>(9_ui8));

  auto stream = GetStream<InputType, TransformedType>(training_batches, true);

  OpTester test("RobustScalerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {1_ui8, 3_ui8, 5_ui8, 7_ui8, 9_ui8});
  test.AddOutput<TransformedType>("Output", {5}, {-1.f, -0.5f, 0.f, 0.5f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering) {
  using InputType = uint8_t;
  using TransformedType = float;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(1_ui8),
      NS::TestHelpers::make_vector<InputType>(7_ui8),
      NS::TestHelpers::make_vector<InputType>(5_ui8),
      NS::TestHelpers::make_vector<InputType>(3_ui8),
      NS::TestHelpers::make_vector<InputType>(9_ui8));

  auto stream = GetStream<InputType, TransformedType>(training_batches, false);

  OpTester test("RobustScalerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {1_ui8, 3_ui8, 5_ui8, 7_ui8, 9_ui8});
  test.AddOutput<TransformedType>("Output", {5}, {1.0/4.0, 3./4., 5./4., 7./4., 9./4.});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
