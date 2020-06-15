// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/StringFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream() {
  dft::StringTransformer<T> transformer;
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, StringTransformer_integer_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  auto stream = GetStream<int64_t>();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Input", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"1", "3", "5", "7", "9"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, StringTransformer_double_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  auto stream = GetStream<double>();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<double>("Input", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"1.000000", "3.000000", "5.000000", "7.000000", "9.000000"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, StringTransformer_bool_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  auto stream = GetStream<bool>();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<bool>("Input", {5}, {true, false, false, false, true});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"True", "False", "False", "False", "True"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, StringTransformer_string_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  auto stream = GetStream<std::string>();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<std::string>("Input", {5}, {"ONE", "three", "FIVE", "SeVeN", "NINE"});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"ONE", "three", "FIVE", "SeVeN", "NINE"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
