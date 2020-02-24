// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/../Archive.h"
#include "Featurizers/MissingDummiesFeaturizer.h"

namespace ft = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream() {
  ft::Archive ar;
  ft::Featurizers::MissingDummiesTransformer<T> inst;
  inst.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, MissingDummiesTransformer_float) {
  OpTester test("MissingDummiesTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<float>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<float>("Input", {2}, {2.5f, std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<int8_t>("Output", {2}, {0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, MissingDummiesTransformer_double) {
  OpTester test("MissingDummiesTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<double>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Input", {2}, {2.5, std::numeric_limits<double>::quiet_NaN()});
  test.AddOutput<int8_t>("Output", {2}, {0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, MissingDummiesTransformer_string) {
  OpTester test("MissingDummiesTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<std::string>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {2}, {"hello", ""});
  test.AddOutput<int8_t>("Output", {2}, {0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
}
}
