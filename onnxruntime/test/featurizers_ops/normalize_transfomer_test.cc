// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MaxNormalizeFeaturizer.h"
#include "Featurizers/L1NormalizeFeaturizer.h"
#include "Featurizers/L2NormalizeFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

template <typename T>
using Range = std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>;

namespace onnxruntime {
namespace test {

namespace {
template <typename EstimatorT, typename InputType>
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& training_batches, size_t col_index) {
  NS::AnnotationMapsPtr pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  EstimatorT estimator(pAllColumnAnnotations, col_index);
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, training_batches);
  auto pTransformer = estimator.create_transformer();

  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, Normalize_double_maxnorm) {
  using ValueType = std::double_t;
  using InputType = Range<ValueType>;
  using EstimatorT = NS::Featurizers::MaxNormalizeEstimator<InputType>;

  std::vector<ValueType> row1({7.9, 4.37, 6, 10});

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(InputType(row1.begin(), row1.end())));

  auto stream = GetStream<EstimatorT, InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("NormalizeTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Input", {4}, row1);
  test.AddOutput<double>("Output", {4}, {0.79, 0.437, 0.6, 1.0});
  test.Run();
}

TEST(FeaturizersTests, Normalize_int16_l2_norm) {
  using ValueType = std::int16_t;
  using InputType = Range<ValueType>;
  using EstimatorT = NS::Featurizers::L2NormalizeEstimator<InputType>;

  std::vector<ValueType> row1({4, 1, 2, 2});
  std::vector<ValueType> row2({1, 3, 9, 3});
  std::vector<ValueType> row3({5, 7, 5, 1});

  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(InputType(row1.begin(), row1.end())),
      NS::TestHelpers::make_vector<InputType>(InputType(row2.begin(), row2.end())),
      NS::TestHelpers::make_vector<InputType>(InputType(row3.begin(), row3.end())));

  auto stream = GetStream<EstimatorT, InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("NormalizeTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<int16_t>("Input", {3, 4}, {
      4, 1, 2, 2,
      1, 3, 9, 3,
      5, 7, 5, 1
      });

  test.AddOutput<double>("Output", {3, 4}, {
      0.8, 0.2, 0.4, 0.4, 
      0.1, 0.3, 0.9, 0.3,
      0.5, 0.7, 0.5, 0.1});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
