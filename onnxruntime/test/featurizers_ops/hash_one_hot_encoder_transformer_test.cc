// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/HashOneHotVectorizerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream() {
  NS::Featurizers::HashOneHotVectorizerTransformer<T> hvtransformer(2, 100);
  NS::Archive ar;
  hvtransformer.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, HashOneHotVectorizerTransformer_int8) {
  using Type = int8_t;
  auto stream = GetStream<Type>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("HashOneHotVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<Type>("Input", {1}, {15});

  test.AddOutput<uint64_t>("NumElements", {1}, {100u});
  test.AddOutput<uint8_t>("Value", {1}, {1u});
  test.AddOutput<uint64_t>("Index", {1}, {29u});

  test.Run();
}

TEST(FeaturizersTests, HashOneHotVectorizerTransformer_int32) {
  using Type = int32_t;
  auto stream = GetStream<Type>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("HashOneHotVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<Type>("Input", {1}, {15});

  test.AddOutput<uint64_t>("NumElements", {1}, {100u});
  test.AddOutput<uint8_t>("Value", {1}, {1u});
  test.AddOutput<uint64_t>("Index", {1}, {22u});

  test.Run();
}

TEST(FeaturizersTests, HashOneHotVectorizerTransformer_double) {
  using Type = double;
  auto stream = GetStream<Type>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("HashOneHotVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<Type>("Input", {1}, {15.0});

  test.AddOutput<uint64_t>("NumElements", {1}, {100u});
  test.AddOutput<uint8_t>("Value", {1}, {1u});
  test.AddOutput<uint64_t>("Index", {1}, {99u});

  test.Run();
}

TEST(FeaturizersTests, HashOneHotVectorizerTransformer_string) {
  using Type = std::string;
  auto stream = GetStream<Type>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("HashOneHotVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<Type>("Input", {1}, {"hello"});

  test.AddOutput<uint64_t>("NumElements", {1}, {100u});
  test.AddOutput<uint8_t>("Value", {1}, {1u});
  test.AddOutput<uint64_t>("Index", {1}, {25u});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime