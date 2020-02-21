// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MeanImputerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream(const std::vector<typename NS::Traits<T>::nullable_type>& trainingBatches, size_t col_index) {
  NS::Featurizers::MeanImputerEstimator<T> estimator(NS::CreateTestAnnotationMapsPtr(1), col_index);
  NS::TestHelpers::Train<dft::MeanImputerEstimator<T>>(estimator, trainingBatches);

  auto pTransformer(estimator.create_transformer());
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, MeanImputerEstimator_float) {
  using InputType = float;
  std::vector<float> trainingBatches = {
      10.0f,
      20.0f,
      NS::Traits<float>::CreateNullValue(),
      30.0f,
      40.0f,
      NS::Traits<float>::CreateNullValue()};

  auto stream = GetStream<InputType>(trainingBatches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("MeanImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<float>("Input", {5}, {NS::Traits<float>::CreateNullValue(), 1.0f, 2.0f, 3.0f, NS::Traits<float>::CreateNullValue()});
  test.AddOutput<double>("Output", {5}, {25.0, 1.0, 2.0, 3.0, 25.0});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
