// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MinMaxImputerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream(const std::vector<typename NS::Traits<T>::nullable_type>& training_batches, size_t col_index, bool use_min) {
  NS::Featurizers::MinMaxImputerEstimator<T> estimator(NS::CreateTestAnnotationMapsPtr(1), col_index, use_min);
  NS::TestHelpers::Train<dft::MinMaxImputerEstimator<T>>(estimator, training_batches);

  auto pTransformer(estimator.create_transformer());
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, MinMaxImputerTransformer_min_float) {
  using InputType = float;
  std::vector<InputType> training_batches{
      10.0f,
      20.0f,
      NS::Traits<float>::CreateNullValue(),
      30.0f,
      NS::Traits<float>::CreateNullValue()};

  auto stream = GetStream<InputType>(training_batches, 0, true);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("MinMaxImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<float>("Input", {5}, {NS::Traits<float>::CreateNullValue(), 1.0f, 2.0f, 3.0f, NS::Traits<float>::CreateNullValue()});
  test.AddOutput<float>("Output", {5}, {10.0f, 1.0f, 2.0f, 3.0f, 10.0f});
  test.Run();
}

TEST(FeaturizersTests, MinMaxImputerTransformer_min_string) {
  using InputType = std::string;
  std::vector<nonstd::optional<std::string>> training_batches = {"10",
                                                                "20",
                                                                nonstd::optional<std::string>(),
                                                                "30",
                                                                nonstd::optional<std::string>()};

  auto stream = GetStream<InputType>(training_batches, 0, true);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("MinMaxImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {"", "1", "2", "3", ""});
  test.AddOutput<InputType>("Output", {5}, {"10", "1", "2", "3", "10"});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
