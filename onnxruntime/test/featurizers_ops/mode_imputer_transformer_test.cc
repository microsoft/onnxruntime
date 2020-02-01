// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/ModeImputerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename InputT>
std::vector<uint8_t> GetStream(const std::vector<typename NS::Traits<InputT>::nullable_type>& training_batches, size_t col_index) {
  dft::ModeImputerEstimator<InputT> estimator(NS::CreateTestAnnotationMapsPtr(1), col_index);
  NS::TestHelpers::Train<dft::ModeImputerEstimator<InputT>>(estimator, training_batches);

  auto transformer(estimator.create_transformer());
  NS::Archive ar;
  transformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, ModeImputerTransformer_float) {
  using InputType = float;

  std::vector<float> training_batches = {
      10.0f,
      20.0f,
      NS::Traits<float>::CreateNullValue(),
      30.0f,
      NS::Traits<float>::CreateNullValue(),
      20.0f,
      NS::Traits<float>::CreateNullValue(),
      40.0f};

  auto stream = GetStream<InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("ModeImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {NS::Traits<float>::CreateNullValue(), 1.0f, 2.0f, 3.0f, NS::Traits<float>::CreateNullValue()});
  test.AddOutput<InputType>("Output", {5}, {20.0f, 1.0f, 2.0f, 3.0f, 20.0f});
  test.Run();
}

TEST(FeaturizersTests, ModeImputerTransformer_string) {
  using InputType = std::string;
  std::vector<nonstd::optional<std::string>> training_batches = {
      "10",
      "20",
      nonstd::optional<std::string>(),
      "30",
      nonstd::optional<std::string>(),
      "20",
      nonstd::optional<std::string>(),
      "40"};

  auto stream = GetStream<InputType>(training_batches, 0);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("ModeImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {"", "1", "2", "3", ""});
  test.AddOutput<InputType>("Output", {5}, {"20", "1", "2", "3", "20"});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
