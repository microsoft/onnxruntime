// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/StandardScaleWrapperFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream(const std::vector<std::vector<T>>& training_batches,
                               size_t col_index, bool with_mean, bool with_std) {
  auto pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  dft::StandardScaleWrapperEstimator<T, double> estimator(pAllColumnAnnotations, col_index, with_mean, with_std);
  NS::TestHelpers::Train<dft::StandardScaleWrapperEstimator<T, double>>(estimator, training_batches);
  auto pTransformer(estimator.create_transformer());
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, StandardScaleWrapperFeaturizer_1_int_with_mean_with_std) {
  using InputType = int32_t;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(10));

  auto stream = GetStream<InputType>(training_batches, 0, true, true);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("StandardScaleWrapperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<int32_t>("Input", {1}, {15});
  test.AddOutput<double>("Output", {1}, {5.});
  test.Run();
}

TEST(FeaturizersTests, StandardScaleWrapperFeaturizer_5_ints_with_mean_without_std) {
  using InputType = int32_t;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(1),
      NS::TestHelpers::make_vector<InputType>(3),
      NS::TestHelpers::make_vector<InputType>(5),
      NS::TestHelpers::make_vector<InputType>(7),
      NS::TestHelpers::make_vector<InputType>(9));

  auto stream = GetStream<InputType>(training_batches, 0, true, false);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("StandardScaleWrapperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<int32_t>("Input", {1}, {2});
  test.AddOutput<double>("Output", {1}, {-3.});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime