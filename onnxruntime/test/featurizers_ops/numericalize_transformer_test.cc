// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/NumericalizeFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream(const std::vector<std::vector<T>>& training_batches, size_t col_index) {
  NS::Featurizers::NumericalizeEstimator<T> estimator(NS::CreateTestAnnotationMapsPtr(1), col_index);
  NS::TestHelpers::Train<dft::NumericalizeEstimator<T>>(estimator, training_batches);

  auto pTransformer(estimator.create_transformer());
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, NumericalizeTransformer_uint32_t) {
  using InputType = int32_t;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(10, 20, 10),
      NS::TestHelpers::make_vector<InputType>(30),
      NS::TestHelpers::make_vector<InputType>(10, 10, 11, 15),
      NS::TestHelpers::make_vector<InputType>(18, 8));

  auto stream = GetStream<InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("NumericalizeTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<int32_t>("Input", {5}, {11, 8, 10, 15, 20});
  test.AddOutput<double>("Output", {5}, {2., 0., 1., 3., 5.});
  test.Run();
}

TEST(FeaturizersTests, NumericalizeTransformer_string) {
  using InputType = std::string;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("orange", "apple", "orange",
                                                "grape", "carrot", "carrot",
                                                "peach", "banana", "orange"));

  auto stream = GetStream<InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("NumericalizeTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {3}, {"banana", "grape", "apple"});
  test.AddOutput<double>("Output", {3}, {1., 3., 0.});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
