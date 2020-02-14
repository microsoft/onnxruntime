// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/CatImputerFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../3rdParty/optional.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename InputType, typename TransformedType>
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& traning_batches) {
  using EstimatorT = NS::Featurizers::CatImputerEstimator<TransformedType>;
  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0);
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, traning_batches);
  auto transfomer = estimator.create_transformer();
  NS::Archive ar;
  transfomer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, CategoryImputer_float_values) {
  using InputType = float;
  using TransformedType = float;
  const InputType null = std::numeric_limits<InputType>::quiet_NaN();

  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(10.f, 20.f, null),
      NS::TestHelpers::make_vector<InputType>(10.f, 30.f, null),
      NS::TestHelpers::make_vector<InputType>(10.f, 10.f, null),
      NS::TestHelpers::make_vector<InputType>(11.f, 15.f, null),
      NS::TestHelpers::make_vector<InputType>(18.f, 8.f, null));

  auto stream = GetStream<InputType, TransformedType>(training_batches);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<InputType>("Input", {7}, {5, 8, 20, null, null, null, null});

  // Expected output.
  test.AddOutput<TransformedType>("Output", {7}, {5, 8, 20, 10, 10, 10, 10});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, CategoryImputer_double_values) {
  using InputType = double;
  using TransformedType = double;
  constexpr InputType null = std::numeric_limits<InputType>::quiet_NaN();

  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto training_batches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>(10., 20., null),
      NS::TestHelpers::make_vector<InputType>(10., 30., null),
      NS::TestHelpers::make_vector<InputType>(10., 10., null),
      NS::TestHelpers::make_vector<InputType>(11., 15., null),
      NS::TestHelpers::make_vector<InputType>(18., 8., null));

  auto stream = GetStream<InputType, TransformedType>(training_batches);
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<InputType>("Input", {7}, {5, 8, 20, null, null, null, null});

  // Expected output.
  test.AddOutput<TransformedType>("Output", {7}, {5, 8, 20, 10, 10, 10, 10});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, CategoryImputer_string_values) {
  using InputType = std::string;
  using TransformedType = std::string;
  const nonstd::optional<std::string> null;

  auto training_batches = NS::TestHelpers::make_vector<std::vector<nonstd::optional<std::string>>>(
      NS::TestHelpers::make_vector<nonstd::optional<std::string>>("one", "one", "one", null, null, "two", null, null, "three"));

  auto stream = GetStream<nonstd::optional<std::string>, TransformedType>(training_batches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {4}, {"one", "two", "three", ""});
  test.AddOutput<InputType>("Output", {4}, {"one", "two", "three", "one"});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
