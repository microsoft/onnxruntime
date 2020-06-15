// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MaxAbsScalerFeaturizer.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {
template <typename InputT, typename TransformedT>
void TestWrapper() {
  auto trainingBatches = NS::TestHelpers::make_vector<std::vector<InputT>>(
      NS::TestHelpers::make_vector<InputT>(static_cast<InputT>(-4)),
      NS::TestHelpers::make_vector<InputT>(static_cast<InputT>(3)),
      NS::TestHelpers::make_vector<InputT>(static_cast<InputT>(0)),
      NS::TestHelpers::make_vector<InputT>(static_cast<InputT>(2)),
      NS::TestHelpers::make_vector<InputT>(static_cast<InputT>(-1)));

  using EstimatorT = NS::Featurizers::MaxAbsScalerEstimator<InputT, TransformedT>;
  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0);
  NS::TestHelpers::Train<EstimatorT, InputT>(estimator, trainingBatches);

  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  auto stream = ar.commit();

  OpTester test("MaxAbsScalerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  test.AddInput<InputT>("Input", {5}, {-4, 3, 0, 2, -1});
  test.AddOutput<TransformedT>("Output", {5}, {-1.f, 0.75f, 0.f, 0.5f, -0.25f});
  test.Run();
}
}  // namespace

TEST(FeaturizersTests, MaxAbsScaler_int8_output_float_double) {
  TestWrapper<int8_t, float>();
  TestWrapper<int64_t, double>();
}

TEST(FeaturizersTests, MaxAbsScaler_float_output_float_double) {
  TestWrapper<float, float>();
  TestWrapper<double, double>();
}
}  // namespace test
}  // namespace onnxruntime
